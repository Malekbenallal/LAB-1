# Лабораторная работа №1 — Классификация пород собак на Stanford Dogs Dataset

**Вариант 3:** RegNet + Rotation + RAdam  
**Дисциплина:** Глубокое обучение / нейронные сети  
**Группа:** 449м  
**Выполнили:** Э. М. Акерма, А.-М. Н. Беналлал

В лабораторной работе реализована система многоклассовой классификации изображений собак по породам. В качестве датасета используется **Stanford Dogs Dataset**, основная аугментация по варианту — **RandomRotation**, основной оптимизатор по варианту — **RAdam**. Дополнительно проведено сравнение с оптимизатором Adam и с обучением модели с нуля.



### Adam и RAdam

**Adam** — адаптивный оптимизатор, который использует скользящие средние градиента и квадрата градиента. Он часто применяется в задачах глубокого обучения благодаря устойчивой сходимости.

**RAdam** — модификация Adam, в которой исправляется проблема нестабильной дисперсии адаптивного шага на ранних итерациях. За счет корректирующего множителя RAdam может вести себя стабильнее в начале обучения.

### Метрики качества

Для оценки качества использованы следующие метрики:

- **Accuracy** — доля правильных предсказаний среди всех объектов.
- **Precision** — доля корректных объектов среди всех объектов, отнесенных моделью к данному классу.
- **Recall** — доля найденных объектов класса среди всех реальных объектов этого класса.
- **F1-score** — гармоническое среднее Precision и Recall.
- **Loss** — значение функции потерь на тестовой выборке.

Precision, Recall и F1 рассчитаны в **macro-усреднении**: метрика сначала считается отдельно для каждого класса, затем усредняется по всем классам. Такой подход важен для многоклассовой классификации, потому что учитывает качество по каждому классу, а не только общую долю правильных ответов.




Ниже представлены сводные результаты трёх экспериментов: Pretrained Adam, Pretrained RAdam и Scratch Adam. В работе использовалась самописная архитектура ManualRegNetY400MF, датасет Stanford Dogs Dataset, а также разделение на обучающую, валидационную и тестовую выборки в пропорции 70/15/15.

## Итоговая таблица результатов

Датасет разделен стратифицированно, то есть изображения каждого класса попали в обучающую, валидационную и тестовую выборки пропорционально.

| Эксперимент | Accuracy | Precision | Recall | F1 | Loss | Предобучение |
|---|---:|
| Pretrained Adam | 0.7771| 0.7965 | 0.7731 | 0.7723 | 0.6995 | Да |
| Pretrained RAdam | 0.7666 | 0.7762 | 0.7619 | 0.7604 | 0.7529 | Да |
| Scratch Adam | 0.0796 | 0.0627 | 0.0753 | 0.0547 | 4.1125 | Нет |

Лучший результат показал эксперимент «Pretrained Adam»: Accuracy = 0.7771, Precision = 0.7965, Recall = 0.7731, F1 = 0.7723. Эксперимент «Pretrained RAdam» дал близкий результат: F1 = 0.7604. Наиболее слабый результат получен у «Scratch Adam»: F1 = 0.0547.

<img width="1365" height="257" alt="image" src="https://github.com/user-attachments/assets/abd764d3-2a64-4111-84df-580be5de16c5" />

Сравнение итоговых метрик Accuracy, Precision, Recall и F1 по всем экспериментам

<img width="1228" height="686" alt="image" src="https://github.com/user-attachments/assets/6bb63ece-25de-4295-98ee-735ae7b6bd83" />

Сравнение итоговых значений F1-score

<img width="1146" height="630" alt="image" src="https://github.com/user-attachments/assets/90513b10-f9ce-49e0-ad37-b7fbdbd1f0cd" />

Сравнение итоговой функции потерь Loss

<img width="1228" height="529" alt="image" src="https://github.com/user-attachments/assets/78c3a887-261e-49c3-9edb-b3c4f55b1d13" />

Кривые обучения для эксперимента Pretrained Adam

<img width="1238" height="549" alt="image" src="https://github.com/user-attachments/assets/d6d6353b-6778-4827-a310-940c19f984d4" />

Кривые обучения для эксперимента Pretrained RAdam

<img width="1251" height="561" alt="image" src="https://github.com/user-attachments/assets/dcb9517d-3b65-4386-8e7e-721f9fc51ebc" />

Кривые обучения для эксперимента Scratch Adam

<img width="1205" height="558" alt="image" src="https://github.com/user-attachments/assets/3c4f2fad-223c-4bc0-91dc-4bcf687773c8" />


## Краткий анализ результатов

Предобученные модели значительно превосходят модель, обученную с нуля. Это подтверждает эффективность transfer learning для задачи классификации пород собак.
Эксперимент Pretrained Adam оказался лучшим по всем ключевым метрикам и достиг F1 = 0.7723.
Эксперимент Pretrained RAdam показал близкий результат: F1 = 0.7604. Разница с Adam небольшая, около 0.0119.
Графики обучения показывают, что предобученные модели быстро набирают качество в первые эпохи и затем выходят на плато. Модель Scratch Adam обучается заметно медленнее и за 6 эпох не успевает выучить качественные признаки.
Итоговый вывод: использование предобученных ImageNet-весов значительно повышает качество классификации на Stanford Dogs Dataset.


## Использованные источники

1. Stanford Dogs Dataset — https://vision.stanford.edu/aditya86/StanfordDogs/
2. Torchvision documentation: RegNet-Y-400MF — https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html
3. PyTorch documentation: Adam optimizer — https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
4. Torchvision documentation: RandomRotation — https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomRotation.html
5. Radosavovic I., Kosaraju R. P., Girshick R., He K., Dollár P. Designing Network Design Spaces. CVPR, 2020.
6. Kingma D. P., Ba J. Adam: A Method for Stochastic Optimization. ICLR, 2015.
7. Liu L. et al. On the Variance of the Adaptive Learning Rate and Beyond. ICLR, 2020.
