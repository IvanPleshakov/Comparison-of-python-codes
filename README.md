# Tinkoff_DL
### Оценка похожести двух программ Python

#### 1. Установка requirements
   
   ```
   pip install -r requirements.txt 
   ```
   
#### 2. Чтобы запустить обучение модели, необходимо ввести в командной строке: 

   ```
   python3 train.py files plagiat1 plagiat2 --model model.pkl
   ```
   
   - files - директория с оригинальными программами <br />
   - plagiat1, plagiat2 - директории с плагиатными программами <br />
   - model.pkl - файл, куда будет сохранена модель
   - примерная длительность обучения - 4 минуты

#### 3. Чтобы оценить похожесть программ:

   ```
   python3 compare.py input.txt scores.txt --model model.pkl
   ```
   
   - input.txt - файл с парами путей к файлам программ, которые нужно проверить на плагиат <br />
   - scores.txt - файл, куда будут сохранены результаты оценки <br />
   - model.pkl - файл с моделью <br />
   
   - пример файла input.txt: <br />
   
   ```
   files/main.py plagiat1/main.py
   files/loss.py plagiat2/loss.py
   files/loss.py files/loss.py
   ```
   - Каждой строке файла input.txt соответствует строка файла scores.txt, в которой записано число от 0 до 1 (процент совпадения программ, т.е. чем ближе число к 1, тем выше вероятность плагиата)
