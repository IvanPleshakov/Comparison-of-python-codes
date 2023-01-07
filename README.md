# Tinkoff_DL
### Оценка похожести двух программ Python

##### 1. Установка requirements

   pip install -r /path/to/requirements.txt
   
##### 2. Чтобы запустить обучение модели, необходимо ввести в командной строке: 

   python3 train.py files plagiat1 plagiat2 --model model.pkl <br />
   
   files - директория с оригинальными программами <br />
   plagiat1, plagiat2 - директории с плагиатными программами <br />
   model.pkl - файл, куда будет сохранена модель.

##### 3. Чтобы оценить похожесть программ:

   python3 compare.py input.txt scores.txt --model model.pkl <br />
   
   input.txt - файл с парами путей программ, которые нужно проверить на плагиат <br />
   scores.txt - файл, куда будут сохранены результаты оценки <br />
   model.pkl - файл с моделью
