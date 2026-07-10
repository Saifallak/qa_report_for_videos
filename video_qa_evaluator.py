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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "ضع_مفتاح_API_الخاص_بك_هنا")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/app/data/input_video.mp4")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "/app/data/qa_report.txt")

# (اختياري) رابط تحميل مباشر للفيديو (مثال: رابط مشاركة SharePoint/OneDrive).
# لو تم ضبط هذا المتغير، سيقوم السكربت بتحميل الفيديو تلقائياً إلى VIDEO_PATH
# قبل البدء في أي معالجة. اتركه فارغاً "" لو الفيديو موجود محلياً بالفعل.
VIDEO_URL = os.environ.get("VIDEO_URL", "")

# (اختياري) مسار ملف كوكيز الجلسة (بصيغة Netscape cookies.txt) المُصدَّر من
# متصفحك بعد تسجيل الدخول، مطلوب فقط لو الرابط محمي بمصادقة (مثل SharePoint
# الخاص بمؤسسة). اتركه فارغاً "" لو الرابط عام ولا يحتاج تسجيل دخول.
COOKIES_FILE = os.environ.get("COOKIES_FILE", "/app/data/cookies.txt")

# اسم الموديل - يفضل موديل يدعم الفيديو الطويل ونافذة سياق كبيرة
# تم اختيار gemini-3.5-flash لأنه أحدث موديل بفري تير فعّال (مش مجرد سعر معلن)
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3.5-flash")

# مسار الصوت المستخرج مؤقتاً (سيتم حذفه تلقائياً بعد الانتهاء)
TEMP_AUDIO_PATH = "/tmp/temp_extracted_audio.mp3"

# وضع معالجة الوسائط (PROCESSING_MODE):
# - "audio": استخراج الصوت ورفعه فقط (توفيراً للتوكنز والوقت). إذا فشل ffmpeg، يفشل السكربت.
# - "video": رفع الفيديو الأصلي مباشرة دون أي محاولة لاستخراج الصوت.
# - "auto": المحاولة الأولى هي استخراج الصوت ورفعه، وإذا فشل الاستخراج أو لم يتوفر ffmpeg يتم رفع الفيديو كاملاً كخطة بديلة (Fallback).
PROCESSING_MODE = os.environ.get("PROCESSING_MODE", "auto").lower()

# مسار ملف تتبع استهلاك الكوتة والتكلفة التراكمية
USAGE_TRACKER_FILE = os.environ.get("USAGE_TRACKER_FILE", "/app/data/gemini_usage_tracker.json")

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


# ==============================================================================
# 3) دالة مساعدة: كتابة رسالة خطأ مفصلة داخل ملف المخرجات بدل توقف السكربت
# ==============================================================================

def write_error_report(error_stage: str, exception: Exception) -> None:
    """
    تكتب تفاصيل أي خطأ يحدث أثناء التنفيذ داخل ملف المخرجات النهائي،
    بدلاً من السماح للسكربت بالتوقف بشكل مفاجئ دون تفسير للمستخدم.
    """
    error_details = (
        f"# ⚠️ تقرير خطأ - فشل تقييم الفيديو\n\n"
        f"**تاريخ ووقت الخطأ:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"**مرحلة حدوث الخطأ:** {error_stage}\n\n"
        f"**نوع الخطأ:** {type(exception).__name__}\n\n"
        f"**رسالة الخطأ:**\n```\n{str(exception)}\n```\n\n"
        f"**تفاصيل تقنية كاملة (Traceback):**\n```\n{traceback.format_exc()}\n```\n"
    )
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(error_details)
        print(f"❌ حدث خطأ أثناء [{error_stage}]. تم توثيق التفاصيل في: {OUTPUT_FILE}")
    except Exception as write_err:
        # كملاذ أخير، اطبع الخطأ في الطرفية إذا تعذرت الكتابة حتى في الملف
        print("فشل حتى في كتابة ملف تقرير الخطأ:", write_err)
        print(error_details)


# ==============================================================================
# 3.5) دالة تحميل الفيديو من رابط مباشر (مثل رابط مشاركة SharePoint) عبر wget
# ==============================================================================

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
# 5) دالة رفع الملف إلى Gemini File API وانتظار اكتمال المعالجة
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
    # ------------------------------------------------------------------------
    # الخطوة 0 (اختيارية): تحميل الفيديو من رابط مباشر (مثل SharePoint) أولاً
    # تعمل فقط لو تم ضبط VIDEO_URL؛ وإلا يتم تخطيها والاعتماد على VIDEO_PATH
    # ------------------------------------------------------------------------
    if VIDEO_URL:
        try:
            download_video_from_url(VIDEO_URL, VIDEO_PATH, COOKIES_FILE)
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
            print("📌 تم اختيار وضع الفيديو فقط (video). سيتم رفع الفيديو كاملاً مباشرة.")
            
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
        # الخطوة 3: رفع الملف إلى Gemini File API وانتظار اكتمال المعالجة
        # ------------------------------------------------------------------
        uploaded_file_ref = upload_and_wait_for_file(client, file_to_upload)

        # ------------------------------------------------------------------
        # الخطوة 4: إرسال طلب التقييم إلى الموديل مع الملف والبرومبت
        # ------------------------------------------------------------------
        print(f"🤖 جارٍ إرسال طلب التحليل إلى الموديل ({MODEL_NAME}) ...")
        print("   (قد تستغرق هذه الخطوة عدة دقائق نظراً لطول الفيديو)")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file_ref, EVALUATION_PROMPT],
        )

        report_text = response.text

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
