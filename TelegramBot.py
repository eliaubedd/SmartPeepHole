from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("Missing required environment variable: TELEGRAM_BOT_TOKEN")

conn = sqlite3.connect("pairings.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS pairings (
    user_id INTEGER PRIMARY KEY,
    raspberry_id TEXT NOT NULL
)""")
#cursor.execute("PRAGMA journal_mode=WAL;")
conn.commit()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Parameters passed after /start
    args = context.args

    if not args:
        await update.message.reply_text(
            "Welcome! Please write /start {code} using code found in the settings of your app to link the bot."
        )
        return

    raspberry_id = args[0]

    print("Pairing request:")
    print("Telegram user:", user_id)
    print("Raspberry ID:", raspberry_id)

    save_pairing(user_id, raspberry_id)

    await update.message.reply_text(
        f"✅ Successfully paired with device {raspberry_id}"
    )

def save_pairing(user_id, raspberry_id):
    print(f"Saving {user_id} -> {raspberry_id}")
    cursor.execute("REPLACE INTO pairings (user_id, raspberry_id) VALUES (?, ?)", (user_id, raspberry_id))
    conn.commit()

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    print("Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()