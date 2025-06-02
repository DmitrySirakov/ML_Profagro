# ML_Profagro

## 🛠️ Стек ML

![Python](https://img.shields.io/badge/-Python_3.10+-090909?style=for-the-badge&logo=python)
![KServe](https://img.shields.io/badge/-KServe-090909?style=for-the-badge&logo=kubernetes)
![Docker](https://img.shields.io/badge/-Docker-090909?style=for-the-badge&logo=docker)
![Jupyter](https://img.shields.io/badge/-Jupyter-090909?style=for-the-badge&logo=jupyter)
![vLLM](https://img.shields.io/badge/-vLLM-090909?style=for-the-badge&logo=cloudsmith)

[Документация проекта на DeepWiki](https://deepwiki.com/DmitrySirakov/ML_Profagro)

## Описание

ML-компонент обеспечивает промышленный деплой и inference моделей машинного обучения и LLM, необходимых для работы гибридного ретривера, реранкера и генерации ответов. Используется KServe для production-сценариев, интеграция с пайплайном Prefect и backend.

## Архитектура

- **kserve-deploy/** — деплой моделей (embedder, reranker, whisper) через KServe, поддержка масштабирования и мониторинга.
- **test.ipynb** — тестирование inference, интеграция с пайплайном.
- **Jupyter-ноутбуки** — эксперименты по оценке качества, генерации синтетических данных (RAGAS).

## Технологический стек

- Python 3.10+
- KServe
- Docker
- Jupyter
- vLLM

## Запуск

### Через Docker
1. Перейдите в папку kserve-deploy
2. Соберите и запустите контейнер:
   ```bash
   docker build -t kserve-deploy .
   docker run -p 8080:8080 kserve-deploy
   ```

### Локально
1. Установите зависимости:
   ```bash
   cd ml/kserve-deploy
   pip install -r requirements.txt
   ```
2. Запустите сервинг:
   ```bash
   python project/serve.py
   ```

## Практическая значимость
- Инференс embedder/реранкер моделей вынесен в отдельные сервисы, что обеспечивает гибкость и масштабируемость пайплайна.
- Использование KServe и vLLM позволяет интегрировать новые модели без остановки основного контура.