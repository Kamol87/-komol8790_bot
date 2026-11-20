import os
import tempfile
import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from google import genai
from groq import Groq

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ADMIN_ID = int(os.environ["ADMIN_ID"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

chat_history = {}
MAX_HISTORY = 10

OWNER_PROMPT = (
    "Ты — личный ИИ-ассистент Джарвис для владельца. "
    "Отвечай четко, по делу, без воды. Понимай русский и узбекский языки. "
    "Помогай структурировать задачи, заметки, расчеты и черновики сообщений/заказов. "
    "Если владелец диктует заметку, заказ или оплату — аккуратно разложи по пунктам "
    "(кто, что, сумма/количество, способ оплаты или срок) и спроси подтверждение перед финальным учетом."
)

CLIENT_PROMPT = (
    "Ты — вежливый и внимательный менеджер по приему заказов и консультациям. "
    "Отвечай на русском или узбекском в зависимости от языка клиента. "
    "Уточняй ассортимент, количество, адрес/время доставки и способ оплаты. "
    "Не выдумывай цены и скидки, которых не знаешь — если чего-то нет в запросе, уточни у клиента "
    "и зафиксируй заявку в понятном списке для менеджера."
)


def get_system_prompt(user_id: int) -> str:
    return OWNER_PROMPT if user_id == ADMIN_ID else CLIENT_PROMPT


def update_history(user_id: int, role: str, text: str):
    history = chat_history.setdefault(user_id, [])
    history.append(f"{role}: {text}")
    if len(history) > MAX_HISTORY:
        chat_history[user_id] = history[-MAX_HISTORY:]


async def ask_gemini(user_id: int, user_text: str) -> str:
    system_prompt = get_system_prompt(user_id)
    update_history(user_id, "Пользователь", user_text)

    context_block = "\n".join(chat_history[user_id])
    full_prompt = f"{system_prompt}\n\nИстория диалога:\n{context_block}\n\nОтвет ассистента:"

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
    )
    reply = (response.text or "Не удалось сформировать ответ.").strip()
    update_history(user_id, "Ассистент", reply)
    return reply


async def transcribe_voice(bot, voice_file_id: str) -> str:
    tg_file = await bot.get_file(voice_file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=True) as tmp:
        await tg_file.download_to_drive(custom_path=tmp.name)
        with open(tmp.name, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(tmp.name, audio_file.read()),
                model="whisper-large-v3-turbo",
            )
    return (transcription.text or "").strip()


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id == ADMIN_ID:
        await update.message.reply_text(
            "Связь установлена. Режим владельца активен — пишите текстом или отправьте голосовое."
        )
    else:
        await update.message.reply_text(
            "Здравствуйте! Напишите ваш вопрос или заказ — я помогу всё оформить."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        reply = await ask_gemini(user_id, text)
    except Exception as e:
        reply = f"Ошибка при обращении к модели: {e}"

    await update.message.reply_text(reply)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    voice = update.message.voice
    if not voice:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        recognized_text = await transcribe_voice(context.bot, voice.file_id)
        if not recognized_text:
            await update.message.reply_text("Не удалось разобрать голос, повторите, пожалуйста.")
            return

        reply = await ask_gemini(user_id, recognized_text)
        if user_id == ADMIN_ID:
            await update.message.reply_text(f"Распознано: «{recognized_text}»\n\n{reply}")
        else:
            await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text(f"Ошибка обработки голосового: {e}")


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("Бот запущен...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
