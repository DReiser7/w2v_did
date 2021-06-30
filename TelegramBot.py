#pip install python-telegram-bot

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging

global model, tokenizer, max_tokens
hate_speech_counter: dict = dict()



def do_telegram_bot():
    updater = Updater('1279015836:AAEQXV5w70Z7fpijHcfL7ACBikuZvRrlWz4', use_context=True)
    dispatcher = updater.dispatcher

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)
    dispatcher.add_handler(echo_handler)

    updater.start_polling()
    updater.idle()


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk in english to me!")


def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text='Thank You for your input.')


if __name__ == '__main__':
    do_telegram_bot()

