from setuptools import setup, find_packages

setup(
    name="OpenModelsHub",
    version="0.3",  # Обновленная версия
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "pandas",  # Для работы с CSV-файлами
        "flask",   # Для создания веб-приложений
        "gym",     # Для работы с игровыми окружениями
        "pygame",
    ],
    description="Library for easy model training, saving, and dataset loading",
    author="YaroPe1",
    author_email="yarpetromack@gmail.com",
    url="openmodelshub.com",  # URL на репозиторий, если есть
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: СС",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
