# ==============================================================================
# Dockerfile - بيئة جاهزة لتشغيل سكربت تقييم جودة الفيديوهات عبر Gemini API
# ==============================================================================
# يحتوي على:
#   - Python 3.11 (خفيفة - slim)
#   - ffmpeg مثبت جاهز (لاستخراج الصوت من الفيديو تلقائياً)
#   - مكتبة google-genai
#
# طريقة البناء:
#   docker build -t video-qa-evaluator .
#
# طريقة التشغيل (مع تمرير مجلد يحتوي الفيديو + استقبال ملف التقرير خارج الحاوية):
#   docker run --rm \
#     -e GEMINI_API_KEY="ضع_مفتاحك_هنا" \
#     -v "$(pwd)/data:/app/data" \
#     video-qa-evaluator
#
# بحيث يكون الفيديو موجوداً داخل: ./data/input_video.mp4
# وسيظهر التقرير الناتج في: ./data/qa_report.txt
# ==============================================================================

FROM python:3.11-slim

# منع Python من إنشاء ملفات .pyc وتفعيل إظهار الـ logs مباشرة بدون buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# تثبيت ffmpeg (لاستخراج الصوت) و wget (لتحميل الفيديو من رابط مباشر مثل SharePoint)
# ثم تنظيف الكاش لتقليل حجم الصورة النهائية
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg wget && \
    rm -rf /var/lib/apt/lists/*

# مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملف المتطلبات أولاً للاستفادة من Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ السكربت الرئيسي
COPY video_qa_evaluator.py .

# مجلد البيانات المشترك (الفيديو المدخل + التقرير المخرج) - يُربط بـ volume عند التشغيل
RUN mkdir -p /app/data

# نقطة التشغيل الافتراضية
CMD ["python", "video_qa_evaluator.py"]
