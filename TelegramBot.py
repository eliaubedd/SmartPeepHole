"""
Telegram Pairing Bot

This module implements a Telegram bot responsible for pairing a Telegram user
with a specific SmartPeephole Raspberry Pi device.

Pairing is performed via the `/start <code>` command, where `<code>` is a device
identifier shown in the mobile app settings. The pairing is persisted locally
in a SQLite database so that the server can later route notifications to the
correct Telegram user(s).

Environment Variables:
    TELEGRAM_BOT_TOKEN (str): Telegram bot token used to authenticate with the
                              Telegram Bot API.

Database:
    pairings.db
        Table: pairings
            - user_id (INTEGER PRIMARY KEY): Telegram user ID.
            - raspberry_id (TEXT NOT NULL): Device identifier (pairing code).
"""

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)
import sqlite3
import os
from dotenv import load_dotenv


# Environment configuration
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("Missing required environment variable: TELEGRAM_BOT_TOKEN")



# Database initialization
# NOTE: check_same_thread=False allows the connection to be used across threads.
conn = sqlite3.connect("pairings.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS pairings (
        user_id INTEGER PRIMARY KEY,
        raspberry_id TEXT NOT NULL
    )
    """
)

conn.commit()



# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the `/start` command and pair a Telegram user with a Raspberry device.

    This handler supports two modes:
        1) `/start` without arguments:
           Shows instructions to the user on how to pair the bot using a device code.
        2) `/start <raspberry_id>`:
           Stores a pairing between the Telegram user's ID and the provided device ID.

    Args:
        update (telegram.Update): Incoming update from Telegram containing message
            metadata (user, chat, message text, etc.).
        context (telegram.ext.ContextTypes.DEFAULT_TYPE): Context object which
            includes arguments (`context.args`) and other runtime state.

    Returns:
        None

    Side Effects:
        - Writes or replaces a row in the local SQLite database (`pairings.db`).
        - Sends a confirmation message to the user in Telegram.

    Notes:
        - `update.effective_user.id` uniquely identifies the Telegram user.
        - The chat ID is retrieved for completeness/debugging but not stored.
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id  # kept for potential future use/logging

    # Arguments passed after /start
    args = context.args

    if not args:
        await update.message.reply_text(
            "Welcome! Please write /start {code} using the code found in the settings "
            "of your app to link the bot."
        )
        return

    raspberry_id = args[0]

    print("Pairing request:")
    print("Telegram user:", user_id)
    print("Chat ID:", chat_id)
    print("Raspberry ID:", raspberry_id)

    save_pairing(user_id, raspberry_id)

    await update.message.reply_text(f"✅ Successfully paired with device {raspberry_id}")



# Persistence utilities
def save_pairing(user_id: int, raspberry_id: str) -> None:
    """
    Persist a pairing between a Telegram user and a Raspberry device.

    This function performs an UPSERT-like operation using `REPLACE INTO`,
    which overwrites any existing pairing for the same `user_id`.

    Args:
        user_id (int): Telegram user ID (primary key in the database).
        raspberry_id (str): Device identifier provided by the user via `/start`.

    Returns:
        None
    """
    print(f"Saving {user_id} -> {raspberry_id}")
    cursor.execute(
        "REPLACE INTO pairings (user_id, raspberry_id) VALUES (?, ?)",
        (user_id, raspberry_id),
    )
    conn.commit()



# Application entry point
def main() -> None:
    """
    Initialize and run the Telegram bot.

    This function:
        - Builds the bot application using the configured token.
        - Registers command handlers.
        - Starts polling Telegram servers for updates (blocking call).

    Returns:
        None

    Raises:
        ValueError: If the bot token is missing (handled earlier at import time).
    """
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))

    print("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
