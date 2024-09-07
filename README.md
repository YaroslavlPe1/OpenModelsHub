# OpenModelsHub

OpenModelsHub — это библиотека для упрощенного обучения моделей машинного обучения, сохранения моделей и загрузки датасетов.

## Установка

Для установки библиотеки используйте команду:

```bash
pip install OpenModelsHub
```
Использование библиотеки
1. Обучение с учителем
Модуль SupervisedLearningModel используется для обучения моделей с учителем (например, классификаторов или регрессоров).

Пример использования:

```python
from OpenModelsHub import SupervisedLearningModel
from sklearn.linear_model import LogisticRegression
from OpenModelsHub.dataset_loader import load_sklearn_dataset

# Загрузка данных
X, y = load_sklearn_dataset('iris')

# Создание и обучение модели
model = SupervisedLearningModel(LogisticRegression())
model.train(X, y)

# Оценка точности модели
print("Точность:", model.evaluate())

# Сохранение модели
model.save_model("supervised_model.pkl")
2. Обучение без учителя
Модуль UnsupervisedLearningModel используется для обучения моделей без учителя (например, кластеризация).
```
Пример использования:

```python
from OpenModelsHub import UnsupervisedLearningModel
from sklearn.datasets import load_iris

# Загрузка данных
X, _ = load_iris(return_X_y=True)

# Создание и обучение модели кластеризации
model = UnsupervisedLearningModel()
model.train(X)

# Предсказание кластеров для данных
clusters = model.predict(X[:5])
print("Кластеры:", clusters)

# Сохранение модели
model.save_model("unsupervised_model.pkl")
3. Обучение с подкреплением
Модуль ReinforcementLearningModel предназначен для обучения с подкреплением. Включает использование Q-обучения для взаимодействия с окружением.
```
Пример использования:

```python
from OpenModelsHub import ReinforcementLearningModel

# Создание модели
rl_model = ReinforcementLearningModel(action_space=3, state_space=5)

# Обучение модели на основе эпизодов
rl_model.train(episodes=100)

# Печать истории потерь
rl_model.print_loss_history()

# Сохранение модели
rl_model.save_model("rl_model.pkl")
Загрузка и сохранение моделей
Вы можете сохранять обученные модели в файлы с помощью метода save_model() и загружать их с помощью метода load_model().
```
Пример:

```python
# Сохранение модели
model.save_model("model.pkl")

# Загрузка модели
model.load_model("model.pkl")

# Предсказания на основе загруженной модели
predictions = model.predict(X[:5])
print("Предсказания:", predictions)
Загрузка датасетов
OpenModels предоставляет два способа загрузки данных для обучения моделей.
```
Загрузка встроенных датасетов из scikit-learn:
```python
from OpenModelsHub.dataset_loader import load_sklearn_dataset

X, y = load_sklearn_dataset('iris')  # Загрузка датасета Iris
Загрузка датасетов из CSV файлов:
```
```python
from OpenModelsHub.dataset_loader import load_csv_dataset

X, y = load_csv_dataset("data.csv")  # Загрузка данных из файла data.csv
```
## Пример использования OpenModels с играми
```python
from OpenModelsHub import GameReinforcementLearningModel
import gym

# Создание игрового окружения (например, CartPole из OpenAI Gym)
env = gym.make('CartPole-v1')
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

# Создание модели
model = GameReinforcementLearningModel(action_space, state_space)

# Обучение агента
model.train(environment=env, episodes=100)

# Тестирование агента
model.play(environment=env, episodes=5)

# Сохранение Q-таблицы
model.save_model("q_table_cartpole.npy")

# Загрузка Q-таблицы
model.load_model("q_table_cartpole.npy")
```

## Справка по библиотеке OpenModels

OpenModelsHub — это библиотека, предназначенная для упрощения процесса обучения моделей машинного обучения, их сохранения и загрузки датасетов. Библиотека предоставляет удобный интерфейс для обучения с учителем, без учителя и обучения с подкреплением.

Основные компоненты библиотеки:
Модели машинного обучения:
```
SupervisedLearningModel: Обучение с учителем.
UnsupervisedLearningModel: Обучение без учителя.
ReinforcementLearningModel: Обучение с подкреплением.
Функции для загрузки датасетов:

load_sklearn_dataset: Загрузка стандартных встроенных датасетов из библиотеки scikit-learn (например, Iris или Digits).
load_csv_dataset: Загрузка датасетов из файлов CSV.
Методы для работы с моделями:

train: Обучение модели.
predict: Прогнозирование на основе обученной модели.
evaluate: Оценка модели (для моделей с учителем).
save_model: Сохранение модели в файл.
load_model: Загрузка модели из файла.
print_loss_history: Печать истории потерь по эпохам (или по итерациям в случае обучения с подкреплением). 
```

```
## Лицензия
Проект OpenModels распространяется под лицензией СС.
```
