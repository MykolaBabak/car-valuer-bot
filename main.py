
import joblib
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
import numpy as np
import os

API_TOKEN = os.getenv("BOT_TOKEN")
MODEL_PATH = 'car_value_model.pkl'

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

class Form(StatesGroup):
    brand = State()
    model = State()
    year = State()
    mileage = State()
    engine = State()
    fuel = State()
    country = State()

@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await message.reply("Привіт! Давай оцінимо твоє авто. Введи марку (наприклад: Toyota):")
    await Form.brand.set()

@dp.message_handler(state=Form.brand)
async def process_brand(message: types.Message, state: FSMContext):
    await state.update_data(brand=message.text)
    await message.reply("Введи модель (наприклад: Corolla):")
    await Form.model.set()

@dp.message_handler(state=Form.model)
async def process_model(message: types.Message, state: FSMContext):
    await state.update_data(model=message.text)
    await message.reply("Рік випуску (наприклад: 2015):")
    await Form.year.set()

@dp.message_handler(state=Form.year)
async def process_year(message: types.Message, state: FSMContext):
    await state.update_data(year=int(message.text))
    await message.reply("Пробіг (в тис. км, наприклад: 150):")
    await Form.mileage.set()

@dp.message_handler(state=Form.mileage)
async def process_mileage(message: types.Message, state: FSMContext):
    await state.update_data(mileage=int(message.text))
    await message.reply("Обʼєм двигуна (л, наприклад: 1.6):")
    await Form.engine.set()

@dp.message_handler(state=Form.engine)
async def process_engine(message: types.Message, state: FSMContext):
    await state.update_data(engine=float(message.text))
    await message.reply("Тип палива (petrol/diesel/hybrid/electric):")
    await Form.fuel.set()

@dp.message_handler(state=Form.fuel)
async def process_fuel(message: types.Message, state: FSMContext):
    await state.update_data(fuel=message.text.lower())
    await message.reply("Країна (UA, EU або USA):")
    await Form.country.set()

@dp.message_handler(state=Form.country)
async def process_country(message: types.Message, state: FSMContext):
    await state.update_data(country=message.text.upper())
    data = await state.get_data()

    model, columns = joblib.load(MODEL_PATH)

    input_dict = {
        'brand': data['brand'],
        'model': data['model'],
        'age': 2025 - data['year'],
        'mileage': data['mileage'],
        'engine': data['engine'],
        'fuel': data['fuel'],
        'country': data['country']
    }

    input_vector = {col: 0 for col in columns}
    for key, value in input_dict.items():
        colname = f"{key}_{value}" if f"{key}_{value}" in columns else key
        if colname in input_vector:
            input_vector[colname] = value if isinstance(value, (int, float)) else 1

    X_input = np.array([list(input_vector.values())])
    price = model.predict(X_input)[0]

    await message.reply(f"Орієнтовна вартість авто: **${round(price, 2)}**")
    await state.finish()

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
