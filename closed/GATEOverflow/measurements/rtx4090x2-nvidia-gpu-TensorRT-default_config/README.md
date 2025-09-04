| Model        | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|--------------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| retinanet    | offline      |   37.319   |     1766.71  | -                 |                                   | passed   |          |
| retinanet    | singlestream |   37.307   |      592.066 | 1.689             |                                   | passed   |          |
| retinanet    | multistream  |   37.336   |     1457.99  | 5.487             |                                   | passed   |          |
| 3d-unet-99.9 | offline      |    0.86112 |        8.321 | -                 |                                   | passed   |          |
| 3d-unet-99.9 | singlestream |    0.86018 |        2.326 | 429.992           |                                   | passed   |          |
| resnet50     | offline      |   76.078   |    88476.4   | -                 |                                   | passed   | passed   |
| resnet50     | singlestream |   76.064   |     3278.69  | 0.305             |                                   | passed   | passed   |
| resnet50     | multistream  |   76.062   |    15904.6   | 0.503             |                                   | passed   | passed   |
| 3d-unet-99   | offline      |    0.86112 |        8.346 | -                 |                                   | passed   |          |
| 3d-unet-99   | singlestream |    0.86018 |        2.332 | 428.835           |                                   | passed   |          |