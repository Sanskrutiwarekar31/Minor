from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ────────────────────────────────────────────────
#                YOUR BOT TOKEN
# ────────────────────────────────────────────────
TOKEN = "8288138279:AAFg3Ql52TFRK-paXr_BCxxBEtu--fDaBYY"

# ────────────────────────────────────────────────
#                START COMMAND
# ────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "🚨 *Welcome to SOS Emergency Bot*\n\n"
        "Your safety matters. This bot helps you send your *live location instantly* "
        "to your emergency contact in case of danger.\n\n"
        "🔴 *How it works:*\n"
        "• Press the SOS button in the app\n"
        "• Your live location is sent immediately\n\n"
        "⚠️ Use only in real emergencies.\n"
        "Stay safe ❤️"
    )

    await update.message.reply_text(
        welcome_text,
        parse_mode="Markdown"
    )


# ────────────────────────────────────────────────
#                   MAIN FUNCTION
# ────────────────────────────────────────────────
def main() -> None:
    # Build the Application
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))

    # Optional: print when bot starts
    print("Telegram SOS Emergency Bot is running... 🚀")

    # Start the bot (polling = long polling)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()