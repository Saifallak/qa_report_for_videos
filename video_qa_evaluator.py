#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
سكربت تقييم جودة (QA) للفيديوهات التعليمية باستخدام Google Gemini API
===============================================================================
الوظيفة:
    - رفع فيديو تعليمي طويل (حتى ساعتين / 300+ ميجابايت) إلى Gemini File API.
    - محاولة استخراج الصوت فقط (MP3) عبر ffmpeg لتقليل التوكنز وزيادة نسبة النجاح،
      وفي حال فشل الاستخراج يتم رفع ملف الفيديو الكامل كخطة بديلة (Fallback).
    - إرسال برومبت تقييم مفصل (جدول معايير من 64 درجة) للموديل.
    - استقبال التقرير، وحفظه في ملف نصي بصيغة Markdown باللغة العربية.
    - حذف الملف من سيرفرات Google بعد الانتهاء لحماية الخصوصية.
    - معالجة كافة الأخطاء (شبكة / API) وتوثيقها داخل ملف المخرجات بدلاً من توقف
      السكربت بشكل مفاجئ.

المتطلبات (Requirements):
    pip install google-genai
    تثبيت ffmpeg على النظام (اختياري، لكن موصى به بشدة):
      - Ubuntu/Debian: sudo apt install ffmpeg
      - Windows: تحميل ffmpeg وإضافته إلى PATH
      - macOS: brew install ffmpeg

طريقة التشغيل:
    python video_qa_evaluator.py
===============================================================================
"""

import os
import sys
import time
import json
import shutil
import tempfile
import subprocess
import traceback
from datetime import datetime

# محاولة تحميل مكتبة python-dotenv لقراءة ملف .env تلقائياً إذا كان موجوداً
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 1) إعدادات المستخدم (المتغيرات القابلة للتعديل)
# ==============================================================================

# يقرأ كل متغير أولاً من Environment Variable (مهم عند التشغيل عبر Docker)،
# وإن لم يكن موجوداً يستخدم القيمة الافتراضية المكتوبة هنا مباشرة.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "ضع_مفتاح_API_الخاص_بك_هنا").strip('"\' ')
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/app/data/input_video.mp4").strip('"\' ')
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "/app/data/qa_report.txt").strip('"\' ')

# (اختياري) رابط تحميل مباشر للفيديو (مثال: رابط مشاركة SharePoint/OneDrive).
# لو تم ضبط هذا المتغير، سيقوم السكربت بتحميل الفيديو تلقائياً إلى VIDEO_PATH
# قبل البدء في أي معالجة. اتركه فارغاً "" لو الفيديو موجود محلياً بالفعل.
VIDEO_URL = os.environ.get("VIDEO_URL", "").strip('"\' ')

# (اختياري) مسار ملف كوكيز الجلسة (بصيغة Netscape cookies.txt) المُصدَّر من
# متصفحك بعد تسجيل الدخول، مطلوب فقط لو الرابط محمي بمصادقة (مثل SharePoint
# الخاص بمؤسسة). اتركه فارغاً "" لو الرابط عام ولا يحتاج تسجيل دخول.
COOKIES_FILE = os.environ.get("COOKIES_FILE", "/app/data/cookies.txt").strip('"\' ')

# اسم الموديل - يفضل موديل يدعم الفيديو الطويل ونافذة سياق كبيرة
# تم اختيار gemini-3.5-flash لأنه أحدث موديل بفري تير فعّال (مش مجرد سعر معلن)
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3.5-flash").strip('"\' ')

# مسار الصوت المستخرج مؤقتاً (سيتم حذفه تلقائياً بعد الانتهاء)
TEMP_AUDIO_PATH = "/tmp/temp_extracted_audio.mp3"

# وضع معالجة الوسائط (PROCESSING_MODE):
# - "audio": استخراج الصوت ورفعه فقط (توفيراً للتوكنز والوقت). إذا فشل ffmpeg، يفشل السكربت.
# - "video": رفع الفيديو الأصلي مباشرة دون أي محاولة لاستخراج الصوت.
# - "auto": المحاولة الأولى هي استخراج الصوت ورفعه، وإذا فشل الاستخراج أو لم يتوفر ffmpeg يتم رفع الفيديو كاملاً كخطة بديلة (Fallback).
PROCESSING_MODE = os.environ.get("PROCESSING_MODE", "auto").lower().strip('"\' ')

# الإجراء الافتراضي عند تجاوز حد التوكنز (TOKEN_LIMIT_ACTION):
# - "1": اكمل رفع الفيديو كاملاً بالرغم من التجاوز
# - "2": قسّم الفيديو إلى أجزاء (45 دقيقة لكل جزء) وادمج النتائج
# - "3": تحويل للصوت فقط (الأسرع والأوفر)
# تُستخدم هذه القيمة كإجابة افتراضية في الوضع التفاعلي، وكإجابة وحيدة في الوضع غير التفاعلي.
TOKEN_LIMIT_ACTION = os.environ.get("TOKEN_LIMIT_ACTION", "3").strip('"\' ')
if TOKEN_LIMIT_ACTION not in ["1", "2", "3"]:
    TOKEN_LIMIT_ACTION = "3"  # fallback آمن

# مسار ملف تتبع استهلاك الكوتة والتكلفة التراكمية
USAGE_TRACKER_FILE = os.environ.get("USAGE_TRACKER_FILE", "/app/data/gemini_usage_tracker.json").strip('"\' ')

# الحد الأقصى لوقت الانتظار (بالثواني) أثناء معالجة الملف على سيرفرات Google
MAX_PROCESSING_WAIT_SECONDS = 1200  # 20 دقيقة كحد أقصى لملف كبير
POLL_INTERVAL_SECONDS = 10


# ==============================================================================
# 2) نص البرومبت (Prompt) - جدول معايير تقييم الفيديو (64 درجة)
# ==============================================================================

EVALUATION_PROMPT = """
أنت خبير تقييم جودة أكاديمي متخصص في تحليل الفيديوهات التعليمية المسجلة لمدربين/معلمين
عبر منصات التعليم عن بُعد. مهمتك مشاهدة الفيديو المرفق بالكامل بعناية فائقة من أوله
حتى آخره (حتى لو كانت مدته تصل إلى ساعتين)، ثم إجراء تقييم جودة (QA) احترافي ودقيق
وفق "جدول معايير تقييم الفيديو" التالي، بإجمالي 64 درجة موزعة كالتالي:

| م | المعيار | الدرجة العظمى |
|---|---------|----------------|
| 1 | الأداء الأكاديمي الشامل (دقة المحتوى، الشرح، تنظيم الأفكار) | 10 |
| 2 | التواصل واللغة (وضوح اللغة، سلامة الأداء اللغوي، الأسلوب) | 8 |
| 3 | تفاعل المدرب (الحيوية، لغة الجسد، نبرة الصوت) | 8 |
| 4 | جودة النطق ورصد الأخطاء (الأخطاء اللغوية أو العلمية إن وجدت) | 6 |
| 5 | التفاعل مع الطلاب والإجابة على أسئلتهم | 5 |
| 6 | الربط والمراجعة بداية السيشن (ربط الجلسة بما سبق) | 5 |
| 7 | التحفيز ومناداة الطلاب بالاسم | 5 |
| 8 | المظهر الاحترافي وفتح الكاميرا | 4 |
| 9 | الهوية البصرية وتفعيل الخلفية الافتراضية (Virtual Background) | 4 |
| 10| بداية الجلسة والترحيب التفاعلي | 4 |
| 11| الالتزام بالوقت الطبيعي للجلسة (من ساعة و45 دقيقة إلى ساعتين) | 3 |
| 12| البيانات التعريفية وذكر كود الجروب | 2 |

**المجموع الكلي = 64 درجة**

--------------------------------------------------------------------------------
تعليمات إلزامية لتنسيق المخرجات:
--------------------------------------------------------------------------------
1. يجب أن يكون الرد بالكامل باللغة العربية الفصحى الواضحة.
2. أنشئ جدول Markdown يحتوي على الأعمدة التالية بالترتيب:
   | المعيار | الدرجة العظمى | الدرجة المستحقة | التبريـر | التوثيق الزمني (Timestamp) |
   - عمود "التوثيق الزمني" يجب أن يذكر الدقيقة والثانية التي لوحظت فيها الملاحظة
     داخل الفيديو بصيغة (HH:MM:SS) أو (MM:SS)، مستندة فعلياً لما شاهدته في الفيديو.
   - إن لم يكن هناك لحظة زمنية محددة (مثل تقييم عام)، اكتب "عام / طوال الجلسة".
3. بعد الجدول مباشرة، اكتب سطر: "**المجموع النهائي = X من 64**" موضحاً فيه X كرقم صحيح
   ناتج عن جمع كل الدرجات المستحقة في الجدول أعلاه بدقة.
4. أضف قسم بعنوان "### نقاط القوة" يسرد أبرز 3-5 نقاط إيجابية لوحظت في أداء المدرب.
5. أضف قسم بعنوان "### التوصيات والتحسينات" يسرد توصيات عملية ومحددة لرفع جودة الأداء
   في الجلسات القادمة، مرتبطة قدر الإمكان بملاحظات موثقة زمنياً من الفيديو نفسه.
6. كن موضوعياً وصارماً في التقييم، ولا تمنح الدرجة الكاملة إلا إذا كان الأداء يستحقها
   فعلياً بناءً على ما شاهدته، وبرر كل درجة بوضوح.
7. لا تضف أي مقدمات أو خواتيم خارج هذا التنسيق (ابدأ مباشرة بالجدول).
"""


# ==============================================================================
# 2.5) هياكل البيانات ودوال تتبع الاستهلاك والتكلفة (Usage & Cost Tracker)
# ==============================================================================

# تسعير الموديلات لكل مليون توكن (دولار أمريكي)
MODEL_PRICING = {
    "gemini-1.5-flash": {
        "input_under_128k": 0.075 / 1_000_000,
        "input_over_128k": 0.15 / 1_000_000,
        "output_under_128k": 0.30 / 1_000_000,
        "output_over_128k": 0.60 / 1_000_000,
    },
    "gemini-1.5-flash-8b": {
        "input_under_128k": 0.0375 / 1_000_000,
        "input_over_128k": 0.075 / 1_000_000,
        "output_under_128k": 0.15 / 1_000_000,
        "output_over_128k": 0.30 / 1_000_000,
    },
    "gemini-1.5-pro": {
        "input_under_128k": 1.25 / 1_000_000,
        "input_over_128k": 2.50 / 1_000_000,
        "output_under_128k": 5.00 / 1_000_000,
        "output_over_128k": 10.00 / 1_000_000,
    },
    "gemini-2.5-flash": {
        "input_under_128k": 0.30 / 1_000_000,
        "input_over_128k": 0.30 / 1_000_000,
        "output_under_128k": 2.50 / 1_000_000,
        "output_over_128k": 2.50 / 1_000_000,
    },
    "gemini-3.5-flash": {
        "input_under_128k": 1.50 / 1_000_000,
        "input_over_128k": 1.50 / 1_000_000,
        "output_under_128k": 9.00 / 1_000_000,
        "output_over_128k": 9.00 / 1_000_000,
    },
}

# حدود الاستهلاك اليومي للمستوى المجاني (Free Tier Limits) كمرجع للمقارنة
FREE_TIER_LIMITS = {
    "gemini-1.5-flash": {"rpd": 1500, "tpm": 1_000_000, "rpm": 15},
    "gemini-1.5-flash-8b": {"rpd": 1500, "tpm": 1_000_000, "rpm": 15},
    "gemini-1.5-pro": {"rpd": 50, "tpm": 32_000, "rpm": 2},
    "gemini-2.5-flash": {"rpd": 1500, "tpm": 1_000_000, "rpm": 15},
    "gemini-3.5-flash": {"rpd": 1500, "tpm": 1_000_000, "rpm": 15},
}

def load_usage_tracker() -> dict:
    """تحميل ملف تتبع الاستهلاك التراكمي."""
    default_data = {
        "daily_reset_date": datetime.now().strftime("%Y-%m-%d"),
        "daily_usage": {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        },
        "lifetime_usage": {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        }
    }
    
    # التأكد من وجود مجلد الوجهة قبل كتابة/قراءة ملف التتبع
    if USAGE_TRACKER_FILE:
        os.makedirs(os.path.dirname(USAGE_TRACKER_FILE) or ".", exist_ok=True)
    
    if not USAGE_TRACKER_FILE or not os.path.exists(USAGE_TRACKER_FILE):
        return default_data
    try:
        with open(USAGE_TRACKER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # التحقق من وجود الحقول الأساسية
            for key in default_data:
                if key not in data:
                    data[key] = default_data[key]
            return data
    except Exception as e:
        print(f"⚠️ فشل قراءة ملف التتبع {USAGE_TRACKER_FILE}، سيتم استخدام إعدادات افتراضية. الخطأ: {e}")
        return default_data

def save_usage_tracker(data: dict) -> None:
    """حفظ ملف تتبع الاستهلاك التراكمي."""
    if not USAGE_TRACKER_FILE:
        return
    try:
        os.makedirs(os.path.dirname(USAGE_TRACKER_FILE) or ".", exist_ok=True)
        with open(USAGE_TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ فشل حفظ ملف التتبع {USAGE_TRACKER_FILE}. الخطأ: {e}")

def get_model_pricing_rates(model_name: str, input_tokens: int, output_tokens: int) -> tuple:
    """استرجاع أسعار المدخلات والمخرجات بناءً على اسم الموديل وحجم السياق."""
    model_key = None
    for key in MODEL_PRICING:
        if key in model_name.lower():
            model_key = key
            break
    if not model_key:
        if "pro" in model_name.lower():
            model_key = "gemini-1.5-pro"
        else:
            model_key = "gemini-3.5-flash"  # افتراضي
            
    pricing = MODEL_PRICING[model_key]
    total_tokens = input_tokens + output_tokens
    
    # في حال تجاوز السياق 128 ألف توكن، تتضاعف أسعار بعض الموديلات
    if total_tokens > 128_000:
        input_rate = pricing.get("input_over_128k", pricing["input_under_128k"])
        output_rate = pricing.get("output_over_128k", pricing["output_under_128k"])
    else:
        input_rate = pricing["input_under_128k"]
        output_rate = pricing["output_under_128k"]
        
    return input_rate, output_rate, model_key

def update_and_report_usage(model_name: str, input_tokens: int, output_tokens: int) -> str:
    """
    تحديث ملف التتبع الإحصائي وطباعة التقرير في الطرفية
    وإرجاع النص لإلحاقه بالتقرير النهائي.
    """
    input_rate, output_rate, model_key = get_model_pricing_rates(model_name, input_tokens, output_tokens)
    
    # حساب تكلفة الطلب الحالي
    current_cost = (input_tokens * input_rate) + (output_tokens * output_rate)
    
    # تحميل وحساب التراكمي
    tracker = load_usage_tracker()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # فحص إعادة تعيين الإحصائيات اليومية
    if tracker.get("daily_reset_date") != today_str:
        tracker["daily_reset_date"] = today_str
        tracker["daily_usage"] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        }
        
    # تحديث إحصائيات اليوم
    tracker["daily_usage"]["requests"] += 1
    tracker["daily_usage"]["input_tokens"] += input_tokens
    tracker["daily_usage"]["output_tokens"] += output_tokens
    tracker["daily_usage"]["cost"] += current_cost
    
    # تحديث الإحصائيات التراكمية مدى الحياة (Lifetime)
    tracker["lifetime_usage"]["requests"] += 1
    tracker["lifetime_usage"]["input_tokens"] += input_tokens
    tracker["lifetime_usage"]["output_tokens"] += output_tokens
    tracker["lifetime_usage"]["cost"] += current_cost
    
    save_usage_tracker(tracker)
    
    # تحديد حدود الكوتة للموديل الحالي
    limits = FREE_TIER_LIMITS.get(model_key, {"rpd": 1500, "tpm": 1_000_000, "rpm": 15})
    max_rpd = limits["rpd"]
    
    spent_rpd = tracker["daily_usage"]["requests"]
    remaining_rpd = max(0, max_rpd - spent_rpd)
    percentage_rpd = (spent_rpd / max_rpd) * 100 if max_rpd > 0 else 0
    
    usage_report = (
        f"\n\n---\n"
        f"### 📊 تقرير استهلاك كوتة Gemini والتكلفة (Token Usage & Cost Report)\n\n"
        f"**1. استهلاك الطلب الحالي (Current Request):**\n"
        f"- **التوكنز المرسلة (Input Tokens):** {input_tokens:,}\n"
        f"- **التوكنز المستلمة (Output Tokens):** {output_tokens:,}\n"
        f"- **إجمالي التوكنز (Total Tokens):** {input_tokens + output_tokens:,}\n"
        f"- **تكلفة الطلب الحالي (Estimated Cost):** `${current_cost:.6f}` USD\n\n"
        f"**2. الاستهلاك اليومي التراكمي ({today_str}):**\n"
        f"- **إجمالي الطلبات اليوم:** {spent_rpd:,} من {max_rpd:,} طلب\n"
        f"- **إجمالي التوكنز اليوم:** {tracker['daily_usage']['input_tokens'] + tracker['daily_usage']['output_tokens']:,} توكن\n"
        f"- **إجمالي التكلفة اليوم:** `${tracker['daily_usage']['cost']:.6f}` USD\n\n"
        f"**3. الكوتة اليومية المتبقية والمستهلكة (Daily Quota Status):**\n"
        f"- **الطلبات المستهلكة (Spent Quota):** {spent_rpd:,} طلب ({percentage_rpd:.2f}%)\n"
        f"- **الطلبات المتبقية اليوم (Remaining Quota):** {remaining_rpd:,} طلب\n"
        f"- *ملاحظة:* الحدود اليومية الافتراضية للحساب المجاني لموديل `{model_key}` هي {max_rpd:,} طلب يومياً.\n\n"
        f"**4. الإحصائيات التراكمية للمشروع (Lifetime Stats):**\n"
        f"- **إجمالي الطلبات:** {tracker['lifetime_usage']['requests']:,} طلب\n"
        f"- **إجمالي التكلفة التراكمية:** `${tracker['lifetime_usage']['cost']:.6f}` USD\n"
    )
    
    # طباعة التقرير في الطرفية بلون مميز
    print("\n" + "=" * 70)
    print("📊 تقرير إحصائيات الاستهلاك والتكلفة الكلية للطلب:")
    print("-" * 70)
    print(f"• توكنز المدخلات: {input_tokens:,} | المخرجات: {output_tokens:,}")
    print(f"• تكلفة هذا الطلب: ${current_cost:.6f} USD")
    print(f"• الطلبات اليومية المستهلكة: {spent_rpd:,} / {max_rpd:,} ({percentage_rpd:.2f}%)")
    print(f"• الطلبات اليومية المتبقية: {remaining_rpd:,}")
    print(f"• التكلفة اليومية الإجمالية: ${tracker['daily_usage']['cost']:.6f} USD")
    print(f"• التكلفة التراكمية الكلية: ${tracker['lifetime_usage']['cost']:.6f} USD")
    print("=" * 70 + "\n")
    
    return usage_report


def get_interactive_input(prompt_text: str, default_val: str) -> str:
    """يطلب المدخلات تفاعلياً من المستخدم مع عرض القيمة الافتراضية."""
    if not sys.stdin.isatty():
        return default_val
    try:
        user_input = input(f"{prompt_text} [{default_val}]: ").strip()
        return user_input if user_input else default_val
    except (EOFError, KeyboardInterrupt):
        print(f"\n⚠️ تم استخدام القيمة الافتراضية: {default_val}")
        return default_val


# ==============================================================================
# 3) دالة مساعدة: كتابة رسالة خطأ مفصلة داخل ملف المخرجات بدل توقف السكربت
# ==============================================================================

def write_error_report(error_stage: str, exception: Exception) -> None:
    """
    تكتب تفاصيل أي خطأ يحدث أثناء التنفيذ داخل ملف المخرجات النهائي،
    بدلاً من السماح للسكربت بالتوقف بشكل مفاجئ دون تفسير للمستخدم.
    """
    # تحميل إحصائيات الكوتة والتكلفة الحالية للمساعدة في التشخيص عند الفشل
    quota_details = ""
    try:
        tracker = load_usage_tracker()
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # تحديد حدود الكوتة للموديل الحالي
        model_key = None
        for key in FREE_TIER_LIMITS:
            if key in MODEL_NAME.lower():
                model_key = key
                break
        if not model_key:
            if "pro" in MODEL_NAME.lower():
                model_key = "gemini-1.5-pro"
            else:
                model_key = "gemini-3.5-flash"
                
        limits = FREE_TIER_LIMITS.get(model_key, {"rpd": 1500, "tpm": 1_000_000, "rpm": 15})
        max_rpd = limits["rpd"]
        
        spent_rpd = tracker["daily_usage"]["requests"]
        remaining_rpd = max(0, max_rpd - spent_rpd)
        percentage_rpd = (spent_rpd / max_rpd) * 100 if max_rpd > 0 else 0
        
        quota_details = (
            f"\n\n---\n"
            f"### 📊 تقرير حالة الكوتة التراكمية عند حدوث الخطأ (Quota Status at Failure)\n\n"
            f"**الإحصائيات اليومية التراكمية ({today_str}):**\n"
            f"- **إجمالي الطلبات المستهلكة اليوم (Spent Quota):** {spent_rpd:,} من {max_rpd:,} طلب ({percentage_rpd:.2f}%)\n"
            f"- **الطلبات اليومية المتبقية (Remaining Quota):** {remaining_rpd:,} طلب\n"
            f"- **إجمالي التكلفة التراكمية اليوم:** `${tracker['daily_usage']['cost']:.6f}` USD\n\n"
            f"**الإحصائيات التراكمية مدى الحياة (Lifetime Stats):**\n"
            f"- **إجمالي الطلبات:** {tracker['lifetime_usage']['requests']:,} طلب\n"
            f"- **إجمالي التكلفة التراكمية:** `${tracker['lifetime_usage']['cost']:.6f}` USD\n"
        )
        
        # طباعة ملخص الكوتة في الطرفية أيضاً عند حدوث خطأ
        print("\n" + "=" * 70)
        print("⚠️ حدث خطأ في التشغيل. حالة الكوتة الحالية:")
        print("-" * 70)
        print(f"• الطلبات اليومية المستهلكة: {spent_rpd:,} / {max_rpd:,} ({percentage_rpd:.2f}%)")
        print(f"• الطلبات اليومية المتبقية: {remaining_rpd:,}")
        print(f"• التكلفة التراكمية الكلية: ${tracker['lifetime_usage']['cost']:.6f} USD")
        print("=" * 70 + "\n")
    except Exception as tracker_err:
        quota_details = f"\n\n*(تعذر تحميل تفاصيل الكوتة: {tracker_err})*"

    # استخراج أي تفاصيل إضافية من استجابة API الخاصة بـ Google GenAI
    api_error_extra = ""
    try:
        if hasattr(exception, "code") and exception.code:
            api_error_extra += f"- **HTTP Status Code:** {exception.code}\n"
        if hasattr(exception, "message") and exception.message:
            api_error_extra += f"- **API Error Message:** {exception.message}\n"
        if hasattr(exception, "status") and exception.status:
            api_error_extra += f"- **API Status Code:** {exception.status}\n"
        
        # في حال وجود أخطاء خام مضمنة (Raw Response JSON)
        if hasattr(exception, "errors") and exception.errors:
            api_error_extra += f"- **Raw Errors Details (errors):**\n```json\n{json.dumps(exception.errors, indent=2, ensure_ascii=False)}\n```\n"
        elif hasattr(exception, "response") and exception.response:
            try:
                res_json = exception.response.json()
                api_error_extra += f"- **Raw API JSON Response (response):**\n```json\n{json.dumps(res_json, indent=2, ensure_ascii=False)}\n```\n"
            except:
                pass
    except Exception as extra_err:
        api_error_extra = f"\n*(تعذر استخراج التفاصيل الإضافية: {extra_err})*"

    error_details = (
        f"# ⚠️ تقرير خطأ - فشل تقييم الفيديو\n\n"
        f"**تاريخ ووقت الخطأ:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"**مرحلة حدوث الخطأ:** {error_stage}\n\n"
        f"**نوع الخطأ:** {type(exception).__name__}\n\n"
        f"**رسالة الخطأ:**\n```\n{str(exception)}\n```\n\n"
        f"**تفاصيل تقنية للخطأ (API Diagnostics):**\n"
        f"{api_error_extra if api_error_extra else 'لا توجد تفاصيل إضافية.'}\n\n"
        f"**تفاصيل تقنية كاملة (Traceback):**\n```\n{traceback.format_exc()}\n```\n"
        f"{quota_details}"
    )
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(error_details)
        print(f"❌ حدث خطأ أثناء [{error_stage}]. تم توثيق التفاصيل في: {OUTPUT_FILE}")
    except Exception as write_err:
        # كملاذ أخير، اطبع الخطأ في الطرفية إذا تعذرت الكتابة حتى في الملف
        print("فشل حتى في كتابة ملف تقرير الخطأ:", write_err)
        print(error_details)


def convert_sharepoint_url_to_download(url: str) -> str:
    """
    تحويل روابط SharePoint الخاصة بالفيديوهات والمستندات إلى روابط تحميل مباشر (Direct Download Link).
    يتم استبدال المعاملات بعد علامة الاستفهام '?' بـ 'download=1'.
    """
    if "sharepoint.com" in url.lower():
        base_url = url.split("?")[0]
        download_url = f"{base_url}?download=1"
        print(f"🔗 تم رصد رابط SharePoint. تم تحويل الرابط تلقائياً إلى رابط تحميل مباشر:\n   {download_url}")
        return download_url
    return url


def download_video_from_url(video_url: str, destination_path: str, cookies_file: str) -> None:
    """
    تحمّل الفيديو من رابط مباشر (مثل رابط SharePoint/OneDrive) باستخدام wget،
    بنفس أسلوب الأمر اللي بيشتغل يدوياً في الترمنال:
        wget --keep-session-cookies "<URL>" -O video.mp4

    - لو ملف الكوكيز (cookies_file) موجود فعلاً، يتم تمريره لـ wget عبر
      --load-cookies للمصادقة على روابط SharePoint المحمية.
    - لو ملف الكوكيز مش موجود، يتم التحميل بدون كوكيز (يفيد في الروابط العامة).
    - يرفع استثناء (Exception) واضح عند فشل التحميل، ليتم التقاطه والتوثيق
      في تقرير الخطأ النهائي بدلاً من توقف السكربت بشكل غامض.
    """
    if not shutil.which("wget"):
        raise EnvironmentError(
            "أداة wget غير مثبتة في هذه البيئة. الرجاء تثبيتها (apt install wget) "
            "أو تحميل الفيديو يدوياً ووضعه في مسار VIDEO_PATH."
        )

    # التأكد من وجود مجلد الوجهة قبل التحميل
    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)

    command = ["wget", "--keep-session-cookies"]

    # إن وُجد ملف كوكيز صالح، أضفه للأمر للمصادقة على الروابط المحمية
    if cookies_file and os.path.exists(cookies_file):
        print(f"🍪 تم العثور على ملف كوكيز، سيتم استخدامه للمصادقة: {cookies_file}")
        command += ["--load-cookies", cookies_file]
    else:
        print("ℹ️ لم يتم العثور على ملف كوكيز، سيتم محاولة التحميل بدونه.")

    command += [video_url, "-O", destination_path]

    print(f"⬇️  جارٍ تحميل الفيديو من الرابط المزوَّد ...")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,  # حد أقصى ساعة كاملة للتحميل (فيديوهات كبيرة/اتصال بطيء)
    )

    stderr_output = result.stderr.decode(errors="ignore")

    # التحقق من نجاح التحميل: كود الخروج صفر + الملف موجود فعلياً وحجمه > 0
    if result.returncode != 0 or not os.path.exists(destination_path) or os.path.getsize(destination_path) == 0:
        raise RuntimeError(
            "فشل تحميل الفيديو عبر wget. تحقق من صلاحية الرابط، أو من صلاحية "
            "وحداثة ملف الكوكيز (قد تكون الجلسة منتهية الصلاحية).\n"
            f"تفاصيل wget:\n{stderr_output[-1000:]}"
        )

    video_size_mb = os.path.getsize(destination_path) / (1024 * 1024)
    print(f"✅ تم تحميل الفيديو بنجاح ({video_size_mb:.2f} MB) في: {destination_path}")


# ==============================================================================
# 4) دالة استخراج الصوت من الفيديو باستخدام ffmpeg (مع Fallback ذكي)
# ==============================================================================

def get_video_duration(video_path: str) -> float:
    """الحصول على طول الفيديو بالثواني باستخدام ffprobe/ffmpeg."""
    import subprocess
    # نتحقق أولاً من توفر ffprobe
    if shutil.which("ffprobe") is None:
        # إذا كان غير متوفر، نحاول استخدام ffmpeg كبديل
        if shutil.which("ffmpeg") is None:
            return 0.0
        try:
            cmd = ["ffmpeg", "-i", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            for line in result.stderr.split("\n"):
                if "Duration" in line:
                    parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return 0.0

    try:
        cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        # محاولة أخيرة بـ ffmpeg
        try:
            cmd = ["ffmpeg", "-i", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            for line in result.stderr.split("\n"):
                if "Duration" in line:
                    parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return 0.0


def split_video_into_segments(video_path: str, output_pattern: str, segment_time_sec: int) -> list:
    """تقسيم الفيديو إلى أجزاء باستخدام ffmpeg دون إعادة ترميز (سريع جداً)."""
    import subprocess
    import glob
    try:
        # التأكد من مسح أي أجزاء قديمة بنفس النمط
        for f in glob.glob(output_pattern.replace("%03d", "*")):
            try:
                os.remove(f)
            except:
                pass
            
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c", "copy", "-map", "0",
            "-f", "segment", "-segment_time", str(segment_time_sec),
            "-reset_timestamps", "1",
            output_pattern
        ]
        print("🎬 جارٍ تقسيم الفيديو إلى أجزاء...")
        subprocess.run(cmd, capture_output=True, check=True)
        # البحث عن الملفات الناتجة وترتيبها
        parts = sorted(glob.glob(output_pattern.replace("%03d", "*")))
        return parts
    except Exception as e:
        print(f"⚠️ فشل تقسيم الفيديو: {e}")
        return []


def is_ffmpeg_available() -> bool:
    """تتحقق من وجود ffmpeg مثبت على النظام ومتاح في PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio_with_ffmpeg(video_path: str, audio_output_path: str) -> bool:
    """
    تستخرج المسار الصوتي فقط من الفيديو وتحوله إلى MP3 عبر ffmpeg.
    - يقلل حجم الملف المرفوع بشكل كبير (توفير توكنز + سرعة رفع أعلى).
    - يعيد True عند النجاح، و False عند أي فشل (ليتم تفعيل خطة الرفع البديلة).
    """
    if not is_ffmpeg_available():
        print("⚠️ ffmpeg غير متوفر على هذا النظام. سيتم تخطي استخراج الصوت.")
        return False

    try:
        print("🎧 جارٍ استخراج الصوت من الفيديو باستخدام ffmpeg ...")
        # الأمر: قراءة الفيديو -> استبعاد الفيديو (-vn) -> ترميز صوت MP3 بجودة جيدة
        command = [
            "ffmpeg",
            "-y",                # الكتابة فوق أي ملف قديم بنفس الاسم بدون سؤال
            "-i", video_path,    # ملف الفيديو المدخل
            "-vn",                # تجاهل تدفق الفيديو (Video None)
            "-acodec", "libmp3lame",
            "-ar", "44100",       # معدل العينة
            "-ab", "128k",        # معدل البت (كافٍ لوضوح الكلام)
            "-ac", "1",           # قناة صوت واحدة (Mono) كافية للتقييم وتقلل الحجم أكثر
            audio_output_path,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,  # حد أقصى 30 دقيقة لعملية الاستخراج
        )

        if result.returncode == 0 and os.path.exists(audio_output_path):
            audio_size_mb = os.path.getsize(audio_output_path) / (1024 * 1024)
            print(f"✅ تم استخراج الصوت بنجاح ({audio_size_mb:.2f} MB).")
            return True
        else:
            stderr_msg = result.stderr.decode(errors="ignore")
            print(f"⚠️ فشل ffmpeg في استخراج الصوت. التفاصيل:\n{stderr_msg[-500:]}")
            return False

    except Exception as e:
        print(f"⚠️ حدث استثناء أثناء استخراج الصوت عبر ffmpeg: {e}")
        return False


# ==============================================================================
# 5) دالة مساعدة: إرسال طلب للموديل مع إعادة المحاولة التلقائية عند الازدحام
# ==============================================================================

def generate_content_with_retry(client, model: str, contents, max_retries: int = 5, base_wait: int = 15):
    """
    يرسل طلب generate_content مع إعادة المحاولة التلقائية عند:
    - 503 UNAVAILABLE (الموديل مشغول مؤقتاً)
    - 429 RESOURCE_EXHAUSTED (تجاوز حد الطلبات في الدقيقة)
    مع انتظار أسي متزايد بين كل محاولة (Exponential Backoff):
    المحاولة 1 → 15s | 2 → 30s | 3 → 60s | 4 → 120s | 5 → 240s
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            err_str = str(e)
            is_retryable = (
                "503" in err_str or "UNAVAILABLE" in err_str or
                "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            )

            if is_retryable and attempt < max_retries:
                wait_sec = base_wait * (2 ** (attempt - 1))  # 15, 30, 60, 120, 240
                print(f"⏳ المحاولة {attempt}/{max_retries} فشلت — ({err_str[:80]}...)")
                print(f"   ⏱️  إعادة المحاولة بعد {wait_sec} ثانية (Exponential Backoff)...")
                time.sleep(wait_sec)
                last_err = e
            else:
                raise e
    raise last_err


def count_tokens_with_retry(client, model: str, contents, max_retries: int = 5, base_wait: int = 15):
    """
    نفس منطق إعادة المحاولة الأسية لكن لطلبات count_tokens.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.count_tokens(model=model, contents=contents)
        except Exception as e:
            err_str = str(e)
            is_retryable = (
                "503" in err_str or "UNAVAILABLE" in err_str or
                "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            )
            if is_retryable and attempt < max_retries:
                wait_sec = base_wait * (2 ** (attempt - 1))
                print(f"⏳ [count_tokens] المحاولة {attempt}/{max_retries} فشلت — إعادة بعد {wait_sec}s...")
                time.sleep(wait_sec)
                last_err = e
            else:
                raise e
    raise last_err



# ==============================================================================
# 6) دالة رفع الملف إلى Gemini File API وانتظار اكتمال المعالجة
# ==============================================================================

def upload_and_wait_for_file(client, file_path: str):
    """
    ترفع الملف (صوت أو فيديو) عبر Gemini File API، وتنتظر حتى تصبح حالته ACTIVE
    (أي جاهز للاستخدام في طلبات generate_content)، مع حد أقصى للانتظار.
    """
    print(f"⬆️  جارٍ رفع الملف: {file_path} ...")
    uploaded_file = client.files.upload(file=file_path)
    print(f"📤 تم بدء الرفع. اسم الملف على السيرفر: {uploaded_file.name}")

    waited_seconds = 0
    # معالجة الملفات الكبيرة تتم في الخلفية على سيرفرات Google؛ لذا ننتظر حتى تصبح ACTIVE
    while uploaded_file.state.name == "PROCESSING":
        if waited_seconds >= MAX_PROCESSING_WAIT_SECONDS:
            raise TimeoutError(
                f"تجاوزت مدة معالجة الملف الحد الأقصى المسموح به "
                f"({MAX_PROCESSING_WAIT_SECONDS} ثانية) دون اكتمال المعالجة."
            )
        print(f"⏳ الملف قيد المعالجة على سيرفرات Google ... "
              f"(مضى {waited_seconds} ثانية)")
        time.sleep(POLL_INTERVAL_SECONDS)
        waited_seconds += POLL_INTERVAL_SECONDS
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError("فشلت معالجة الملف على سيرفرات Google (حالة الملف: FAILED).")

    print("✅ الملف جاهز الآن للاستخدام (ACTIVE).")
    return uploaded_file


# ==============================================================================
# 6) الدالة الرئيسية: تنسيق كل خطوات العملية من البداية للنهاية
# ==============================================================================

def main():
    global VIDEO_URL, PROCESSING_MODE
    split_mode = False
    
    # ------------------------------------------------------------------------
    # إدخال البيانات تفاعلياً (تخطي البرومبت لو تم التشغيل في وضع غير تفاعلي TTY)
    # ------------------------------------------------------------------------
    if sys.stdin.isatty():
        print("\n" + "=" * 70)
        print("💡 إعداد تشغيل السكربت تفاعلياً:")
        print("   (اضغط Enter مباشرة لاعتماد القيمة الافتراضية المعروضة بين الأقواس)")
        print("=" * 70)
        
        VIDEO_URL = get_interactive_input("🔗 أدخل رابط الفيديو (أو رابط SharePoint)", VIDEO_URL)
        
        # التأكد من صحة خيار وضع المعالجة
        while True:
            mode_input = get_interactive_input("🎧 اختر وضع المعالجة (auto / audio / video / audio_fallback)", PROCESSING_MODE).lower()
            if mode_input in ["auto", "audio", "video", "audio_fallback"]:
                PROCESSING_MODE = mode_input
                break
            print("❌ اختيار غير صالح. الرجاء كتابة: auto أو audio أو video أو audio_fallback")
            
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------------
    # الخطوة 0 (اختيارية): تحميل الفيديو من رابط مباشر (مثل SharePoint) أولاً
    # تعمل فقط لو تم ضبط VIDEO_URL؛ وإلا يتم تخطيها والاعتماد على VIDEO_PATH
    # ------------------------------------------------------------------------
    if VIDEO_URL:
        try:
            final_url = convert_sharepoint_url_to_download(VIDEO_URL)
            download_video_from_url(final_url, VIDEO_PATH, COOKIES_FILE)
        except Exception as e:
            write_error_report("تحميل الفيديو من الرابط (VIDEO_URL)", e)
            return  # نوقف التنفيذ هنا لأنه لا يوجد فيديو للمتابعة به

    # --- التحقق الأولي من وجود ملف الفيديو ---
    if not os.path.exists(VIDEO_PATH):
        write_error_report(
            "التحقق من وجود ملف الفيديو",
            FileNotFoundError(f"لم يتم العثور على ملف الفيديو في المسار: {VIDEO_PATH}")
        )
        return

    # --- استيراد مكتبة google-genai (نضعه هنا لنلتقط خطأ عدم التثبيت بشكل منظم) ---
    try:
        from google import genai
    except ImportError as e:
        write_error_report(
            "استيراد مكتبة google-genai",
            ImportError(
                "المكتبة غير مثبتة. الرجاء تشغيل: pip install google-genai\n"
                f"تفاصيل: {e}"
            )
        )
        return

    uploaded_file_ref = None   # سنحتفظ بمرجع الملف المرفوع لحذفه لاحقاً (Cleanup)
    client = None

    try:
        # ------------------------------------------------------------------
        # الخطوة 1: تهيئة عميل Gemini
        # ------------------------------------------------------------------
        client = genai.Client(api_key=GEMINI_API_KEY)

        # طباعة الموديلات المتاحة لمساعدة المستخدم في معرفة الموديل الفعّال لحسابه
        try:
            available_models = [m.name.replace("models/", "") for m in client.models.list()]
            print(f"📋 الموديلات المتاحة لحسابك (API Key): {len(available_models)} موديل")
            for i, m in enumerate(available_models, 1):
                print(f"   {i:>2}. {m}")
        except Exception as list_err:
            print(f"⚠️ تعذر سرد الموديلات المتاحة: {list_err}")

        # ------------------------------------------------------------------
        # الخطوة 2: تحديد ملف الرفع بناءً على وضع المعالجة (PROCESSING_MODE)
        # ------------------------------------------------------------------
        file_to_upload = None
        
        if PROCESSING_MODE == "audio":
            audio_extracted = extract_audio_with_ffmpeg(VIDEO_PATH, TEMP_AUDIO_PATH)
            if not audio_extracted:
                raise RuntimeError("فشل استخراج الصوت من الفيديو باستخدام ffmpeg، والوضع المختار هو استخراج الصوت فقط (audio).")
            file_to_upload = TEMP_AUDIO_PATH
            print("📌 تم اختيار وضع الصوت فقط (audio). تم استخراج الصوت بنجاح وسيتم رفعه.")
            
        elif PROCESSING_MODE == "video":
            file_to_upload = VIDEO_PATH
            print("📌 تم اختيار وضع الفيديو فقط (video). سيتم رفع الفيديو كاملاً مباشرة (دون تراجع للصوت).")

        elif PROCESSING_MODE == "audio_fallback":
            file_to_upload = VIDEO_PATH
            print("📌 تم اختيار وضع الفيديو مع إمكانية الارتداد للصوت (audio_fallback). سيتم رفع الفيديو أولاً.")
            
        else: # auto
            print("📌 تم اختيار الوضع التلقائي (auto). محاولة استخراج الصوت كخيار أول...")
            audio_extracted = extract_audio_with_ffmpeg(VIDEO_PATH, TEMP_AUDIO_PATH)
            if audio_extracted:
                file_to_upload = TEMP_AUDIO_PATH
                print("   ✅ نجح استخراج الصوت. سيتم رفع الملف الصوتي المستخرج.")
            else:
                file_to_upload = VIDEO_PATH
                print("   ⚠️ فشل استخراج الصوت. سيتم رفع ملف الفيديو الأصلي كاملاً (Fallback).")

        # ------------------------------------------------------------------
        # الخطوة 2.5: تقدير عدد التوكنز مسبقاً قبل الرفع لتوفير وقت وموارد المستخدم
        # ------------------------------------------------------------------
        duration_sec = get_video_duration(file_to_upload)
        if duration_sec > 0:
            if file_to_upload == TEMP_AUDIO_PATH:
                # الصوت يستهلك حوالي 80 توكن لكل ثانية
                estimated_tokens = int(duration_sec * 80) + 5000
                print(f"📊 التوكنز المقدرة للملف الصوتي: {estimated_tokens:,} توكن (الحد الأقصى للموديل: 1,048,576)")
            else:
                # الفيديو يستهلك 258 توكن لكل ثانية (إطار واحد لكل ثانية)
                estimated_tokens = int(duration_sec * 258) + 5000
                print(f"📊 التوكنز المقدرة لملف الفيديو: {estimated_tokens:,} توكن (الحد الأقصى للموديل: 1,048,576)")
                
                # إذا كانت التوكنز تتخطى 1 مليون، ونحن في وضع يسمح بالتحويل للصوت (auto أو audio_fallback) أو التقسيم
                if estimated_tokens > 1_000_000:
                    print(f"⚠️ تنبيه: الفيديو يتخطى الحد الأقصى للموديل ({estimated_tokens:,} > 1,000,000 توكن).")
                    
                    choice = TOKEN_LIMIT_ACTION  # الافتراضي من .env
                    if sys.stdin.isatty():
                        print("\nالرجاء اختيار أحد الخيارات التالية لتفادي فشل العملية:")
                        print("1 - اكمل رفع الفيديو ؟")
                        print("2 - ااقسمه علي فيديوهات تقريبا كل فيديو بحد اقصي ٧٠٠ الف (كل جزء 45 دقيقة)")
                        print("3 - نروح للصوت وبس بقي")
                        
                        while True:
                            choice = get_interactive_input("❓ أدخل رقم الخيار المناسب [1-3]", TOKEN_LIMIT_ACTION).strip()
                            if choice in ["1", "2", "3"]:
                                break
                            print("❌ إدخال غير صالح. الرجاء إدخال 1 أو 2 أو 3")

                    if choice == "2":
                        split_mode = True
                        print("📌 تم اختيار تقسيم الفيديو إلى أجزاء...")
                    elif choice == "3":
                        print("📌 تم اختيار التحويل لمعالجة الصوت فقط...")
                        # استخراج الصوت وتحويل مسار الرفع إليه
                        audio_extracted = extract_audio_with_ffmpeg(VIDEO_PATH, TEMP_AUDIO_PATH)
                        if audio_extracted:
                            file_to_upload = TEMP_AUDIO_PATH
                            duration_sec = get_video_duration(file_to_upload)
                            estimated_tokens = int(duration_sec * 80) + 5000
                            print(f"📊 التوكنز المقدرة لملف الصوت الجديد: {estimated_tokens:,} توكن.")
                        else:
                            print("⚠️ تعذر استخراج الصوت، سيتم محاولة رفع الفيديو بالرغم من كبر حجمه.")
                    else:
                        print("⚠️ سيتم المتابعة ورفع ملف الفيديو الكامل بالرغم من تجاوز الحد الأقصى المتوقع للتوكنز.")

        if split_mode:
            # تقسيم الفيديو إلى أجزاء (كل جزء 2700 ثانية = 45 دقيقة)
            _, ext = os.path.splitext(VIDEO_PATH)
            output_pattern = os.path.join(tempfile.gettempdir(), f"part_%03d{ext}")
            parts = split_video_into_segments(VIDEO_PATH, output_pattern, 2700)
            
            if not parts:
                raise RuntimeError("تعذر تقسيم الفيديو إلى أجزاء.")
                
            print(f"✅ تم تقسيم الفيديو بنجاح إلى {len(parts)} أجزاء.")
            
            # معالجة الأجزاء جزءاً جزءاً
            part_reports = []
            for idx, part_path in enumerate(parts):
                print("\n" + "-" * 70)
                print(f"📤 معالجة الجزء {idx+1} من {len(parts)} ({os.path.basename(part_path)}) ...")
                print("-" * 70)
                
                # رفع الجزء
                uploaded_part_ref = upload_and_wait_for_file(client, part_path)
                
                # حساب التوكنز الفعلية للجزء
                try:
                    token_count_resp = count_tokens_with_retry(
                        client, MODEL_NAME,
                        [uploaded_part_ref, EVALUATION_PROMPT]
                    )
                    print(f"🎯 التوكنز الفعلية للجزء {idx+1}: {token_count_resp.total_tokens:,} توكن.")
                except Exception as cnt_err:
                    print(f"⚠️ تعذر حساب التوكنز للجزء: {cnt_err}")
                
                # إرسال طلب التحليل للجزء
                print(f"🤖 تحليل الجزء {idx+1}...")
                part_response = generate_content_with_retry(
                    client, MODEL_NAME,
                    [uploaded_part_ref, EVALUATION_PROMPT],
                )
                part_text = part_response.text
                if not part_text or not part_text.strip():
                    raise ValueError(f"رد الموديل للجزء {idx+1} جاء فارغاً.")
                    
                part_reports.append(part_text)
                
                # تسجيل الكوتة والتكلفة للجزء
                input_tokens = 0
                output_tokens = 0
                if part_response.usage_metadata:
                    input_tokens = part_response.usage_metadata.prompt_token_count or 0
                    output_tokens = part_response.usage_metadata.candidates_token_count or 0
                update_and_report_usage(MODEL_NAME, input_tokens, output_tokens)
                
                # تنظيف خادم Google والملف المحلي للجزء
                try:
                    client.files.delete(name=uploaded_part_ref.name)
                    print(f"🧹 تم حذف ملف الجزء {idx+1} من سيرفرات Google.")
                except Exception as del_err:
                    print(f"⚠️ تعذر حذف ملف الجزء {idx+1}: {del_err}")
                try:
                    os.remove(part_path)
                except:
                    pass
            
            # تصفير المرجع لتجنب حذفه مرة أخرى في finally
            uploaded_file_ref = None
            
            # دمج التقارير
            print("\n" + "=" * 70)
            print("🤖 جارٍ دمج التقارير المجزأة في تقرير نهائي متناسق...")
            print("=" * 70)
            
            merge_prompt = (
                "أنت مسؤول جودة تعليمية خبير. لديك أدناه تقارير تقييم جودة مجزأة لمحاضرة واحدة تم تقسيمها إلى أجزاء.\n"
                "المطلوب منك هو دمج هذه التقارير في تقرير واحد شامل ومتناسق يتبع نفس الهيكل والجدول والدرجة النهائية (من 64).\n"
                "قم بدمج الملاحظات ونقاط القوة والضعف وجدول التقييم والدرجات بشكل منطقي ومتسق.\n\n"
                "التقارير المجزأة هي:\n\n"
            )
            for idx, r_text in enumerate(part_reports):
                merge_prompt += f"--- تقرير الجزء {idx+1} ---\n{r_text}\n\n"
                
            merge_response = generate_content_with_retry(
                client, MODEL_NAME,
                [merge_prompt],
            )
            report_text = merge_response.text
            
            if not report_text or not report_text.strip():
                raise ValueError("فشل الموديل في دمج التقارير وجاء الرد فارغاً.")
                
            # تسجيل كوتة الدمج
            input_tokens = 0
            output_tokens = 0
            if merge_response.usage_metadata:
                input_tokens = merge_response.usage_metadata.prompt_token_count or 0
                output_tokens = merge_response.usage_metadata.candidates_token_count or 0
            usage_report_markdown = update_and_report_usage(MODEL_NAME, input_tokens, output_tokens)
            
        else:
            # ------------------------------------------------------------------
            # الخطوة 3: رفع الملف إلى Gemini File API وانتظار اكتمال المعالجة
            # ------------------------------------------------------------------
            uploaded_file_ref = upload_and_wait_for_file(client, file_to_upload)

            # حساب التوكنز الفعلية قبل إرسال الطلب للموديل لتوفير معلومات دقيقة للمستخدم
            try:
                print("📊 جارٍ حساب التوكنز الفعلية للطلب...")
                token_count_resp = count_tokens_with_retry(
                    client, MODEL_NAME,
                    [uploaded_file_ref, EVALUATION_PROMPT]
                )
                exact_tokens = token_count_resp.total_tokens
                print(f"🎯 التوكنز الفعلية للطلب (الملف + البرومبت): {exact_tokens:,} توكن.")
                if exact_tokens > 1_048_576:
                    print(f"⚠️ تحذير: التوكنز الفعلية تتجاوز حد الموديل (1,048,576). قد يفشل الطلب.")
            except Exception as cnt_err:
                print(f"⚠️ تعذر حساب التوكنز الفعلية بدقة: {cnt_err}")

            # ------------------------------------------------------------------
            # الخطوة 4: إرسال طلب التقييم إلى الموديل مع الملف والبرومبت
            # ------------------------------------------------------------------
            print(f"🤖 جارٍ إرسال طلب التحليل إلى الموديل ({MODEL_NAME}) ...")
            print("   (قد تستغرق هذه الخطوة عدة دقائق نظراً لطول الفيديو)")

            try:
                response = generate_content_with_retry(
                    client, MODEL_NAME,
                    [uploaded_file_ref, EVALUATION_PROMPT],
                )
                report_text = response.text
            except Exception as api_err:
                # إذا كان الملف الذي تم رفعه هو الفيديو الكامل ووضع المعالجة هو audio_fallback، نحاول استخراج الصوت كبديل تلقائي
                if file_to_upload == VIDEO_PATH and PROCESSING_MODE == "audio_fallback":
                    print(f"⚠️ فشل التحليل باستخدام الفيديو الكامل: {api_err}")
                    print("📌 محاولة استخراج الصوت ورفعه كخيار بديل لتفادي المشكلة...")
                    
                    audio_extracted = extract_audio_with_ffmpeg(VIDEO_PATH, TEMP_AUDIO_PATH)
                    if audio_extracted:
                        try:
                            # حذف ملف الفيديو القديم من سيرفرات Google لتوفير المساحة والخصوصية
                            try:
                                client.files.delete(name=uploaded_file_ref.name)
                                print("🧹 تم حذف ملف الفيديو القديم من سيرفرات Google.")
                            except Exception as del_err:
                                print(f"⚠️ تعذر حذف ملف الفيديو القديم: {del_err}")
                                
                            # رفع ملف الصوت الجديد
                            uploaded_file_ref = upload_and_wait_for_file(client, TEMP_AUDIO_PATH)
                            print("🤖 إعادة إرسال طلب التحليل إلى الموديل باستخدام ملف الصوت البديل...")
                            response = generate_content_with_retry(
                                client, MODEL_NAME,
                                [uploaded_file_ref, EVALUATION_PROMPT],
                            )
                            report_text = response.text
                            # تحديث مسار الملف المرفوع لضمان تنظيفه في كتلة finally
                            file_to_upload = TEMP_AUDIO_PATH
                        except Exception as fallback_err:
                            raise RuntimeError(f"فشلت المحاولة البديلة أيضاً باستخدام الصوت.\nخطأ محاولة الصوت: {fallback_err}\nخطأ محاولة الفيديو الأصلية: {api_err}")
                    else:
                        raise RuntimeError(f"فشل التحليل باستخدام الفيديو الأصلي ({api_err})، ولم يتوفر ffmpeg أو تعذر استخراج الصوت كخيار بديل.")
                else:
                    raise api_err

            if not report_text or not report_text.strip():
                raise ValueError("رد الموديل جاء فارغاً دون أي محتوى نصي.")

            # الحصول على بيانات الاستهلاك (usage_metadata)
            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0

            # تحديث وحساب الاستهلاك والتكلفة والكوتة
            usage_report_markdown = update_and_report_usage(MODEL_NAME, input_tokens, output_tokens)

        # ------------------------------------------------------------------
        # الخطوة 5: حفظ التقرير النهائي في ملف المخرجات
        # ------------------------------------------------------------------
        final_report = (
            f"# تقرير تقييم جودة الفيديو التعليمي (QA Evaluation Report)\n\n"
            f"**اسم ملف الفيديو:** {os.path.basename(VIDEO_PATH)}\n"
            f"**تاريخ التقييم:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**الموديل المستخدم:** {MODEL_NAME}\n"
            f"**وضع المعالجة:** {PROCESSING_MODE}\n\n"
            f"---\n\n"
            f"{report_text}"
            f"{usage_report_markdown}"
        )

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(final_report)

        print(f"🎉 تم إنشاء التقرير بنجاح وحفظه في: {OUTPUT_FILE}")

    except Exception as e:
        # أي خطأ يحدث في أي خطوة أعلاه (شبكة، API، حجم ملف، ...) يُلتقط هنا
        # ويُكتب بالتفصيل في ملف المخرجات بدلاً من إيقاف السكربت بشكل مفاجئ.
        write_error_report("أثناء عملية الرفع أو التحليل عبر Gemini API", e)

    finally:
        # ------------------------------------------------------------------
        # الخطوة 6: التنظيف (Cleanup) - حذف الملف من سيرفرات Google للخصوصية
        # ------------------------------------------------------------------
        if client is not None and uploaded_file_ref is not None:
            try:
                print("🧹 جارٍ حذف الملف من سيرفرات Google المؤقتة (Cleanup)...")
                client.files.delete(name=uploaded_file_ref.name)
                print("✅ تم حذف الملف من سيرفرات Google بنجاح.")
            except Exception as cleanup_error:
                # فشل الحذف ليس خطأً حرجاً، لكن نوثقه في الطرفية فقط
                print(f"⚠️ تعذر حذف الملف من سيرفرات Google: {cleanup_error}")

        # حذف ملف الصوت المؤقت المحلي إن وُجد
        if os.path.exists(TEMP_AUDIO_PATH):
            try:
                os.remove(TEMP_AUDIO_PATH)
                print("🧹 تم حذف ملف الصوت المؤقت المحلي.")
            except Exception as local_cleanup_error:
                print(f"⚠️ تعذر حذف ملف الصوت المؤقت المحلي: {local_cleanup_error}")


# ==============================================================================
# نقطة انطلاق السكربت
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 بدء تشغيل سكربت تقييم جودة الفيديو التعليمي عبر Gemini API")
    print("=" * 70)
    main()
    print("=" * 70)
    print("🏁 انتهى تنفيذ السكربت.")
    print("=" * 70)
