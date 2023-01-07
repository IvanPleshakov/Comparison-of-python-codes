# Tinkoff_DL
### Оценка похожести двух программ Python

1. Установка requirements

   pip install -r /path/to/requirements.txt
   
2. Чтобы запустить обучение модели, необходимо ввести в командной строке: 
   python3 train.py files plagiat1 plagiat2 --model model.pkl \n
   files - директория с оригинальными программами
   plagiat1, plagiat2 - директории с плагиатными программами
   model.pkl - файл, куда будет сохранена модель.

3. Чтобы оценить похожесть программ:
   python3 compare.py input.txt scores.txt --model model.pkl 
   input.txt - файл с парами путей программ, которые нужно проверить на плагиат
   scores.txt - файл, куда будут сохранены результаты оценки
   model.pkl - файл с моделью
