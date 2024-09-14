# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/knakamura13/mlrose-ky/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                             |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/mlrose\_ky/\_\_init\_\_.py                                   |       19 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/\_\_init\_\_.py                        |        9 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/crossovers/\_\_init\_\_.py             |        3 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/crossovers/\_crossover\_base.py        |       11 |        1 |     91% |        73 |
| src/mlrose\_ky/algorithms/crossovers/one\_point\_crossover.py    |        9 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/crossovers/tsp\_crossover.py           |       48 |       31 |     35% |86, 111-149 |
| src/mlrose\_ky/algorithms/crossovers/uniform\_crossover.py       |       10 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/decay/\_\_init\_\_.py                  |        4 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/decay/arith\_decay.py                  |       27 |       13 |     52% |50, 52, 54, 57, 60, 63-65, 100-112 |
| src/mlrose\_ky/algorithms/decay/custom\_schedule.py              |       21 |       10 |     52% |37, 40, 43-45, 79-90 |
| src/mlrose\_ky/algorithms/decay/expl\_decay.py                   |       28 |       13 |     54% |52, 54, 56, 59, 62, 65-67, 101-113 |
| src/mlrose\_ky/algorithms/decay/geom\_decay.py                   |       27 |       13 |     52% |50, 52, 54, 57, 60, 63-65, 99-111 |
| src/mlrose\_ky/algorithms/ga.py                                  |       87 |       22 |     75% |29, 48, 79-91, 209-213, 221-222, 253-254, 288-289 |
| src/mlrose\_ky/algorithms/gd.py                                  |       40 |        6 |     85% |73, 82, 111, 115-116, 127 |
| src/mlrose\_ky/algorithms/hc.py                                  |       47 |       10 |     79% |98-100, 113, 117-129, 143-144, 148 |
| src/mlrose\_ky/algorithms/mimic.py                               |       42 |        1 |     98% |        92 |
| src/mlrose\_ky/algorithms/mutators/\_\_init\_\_.py               |        4 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/mutators/\_mutator\_base.py            |       11 |        1 |     91% |        64 |
| src/mlrose\_ky/algorithms/mutators/change\_one\_mutator.py       |       14 |        1 |     93% |        61 |
| src/mlrose\_ky/algorithms/mutators/discrete\_mutator.py          |       17 |       11 |     35% |34-36, 57-68 |
| src/mlrose\_ky/algorithms/mutators/shift\_one\_mutator.py        |       14 |        8 |     43% |34-36, 56-62 |
| src/mlrose\_ky/algorithms/mutators/swap\_mutator.py              |       11 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/rhc.py                                 |       54 |       12 |     78% |106-108, 136-139, 143-156, 164, 168 |
| src/mlrose\_ky/algorithms/sa.py                                  |       42 |        5 |     88% |95, 112, 136-137, 150 |
| src/mlrose\_ky/decorators/\_\_init\_\_.py                        |        1 |        0 |    100% |           |
| src/mlrose\_ky/decorators/short\_name\_decorator.py              |       12 |        0 |    100% |           |
| src/mlrose\_ky/fitness/\_\_init\_\_.py                           |       10 |        0 |    100% |           |
| src/mlrose\_ky/fitness/\_discrete\_peaks\_base.py                |       14 |        2 |     86% |    38, 66 |
| src/mlrose\_ky/fitness/continuous\_peaks.py                      |       32 |        3 |     91% |45, 104, 123 |
| src/mlrose\_ky/fitness/custom\_fitness.py                        |       15 |        3 |     80% |60, 85, 97 |
| src/mlrose\_ky/fitness/flip\_flop.py                             |       18 |        2 |     89% |    52, 77 |
| src/mlrose\_ky/fitness/four\_peaks.py                            |       19 |        2 |     89% |    65, 86 |
| src/mlrose\_ky/fitness/knapsack.py                               |       30 |        6 |     80% |72, 74, 76, 78, 104, 106 |
| src/mlrose\_ky/fitness/max\_k\_color.py                          |       18 |        1 |     94% |        92 |
| src/mlrose\_ky/fitness/one\_max.py                               |       11 |        1 |     91% |        53 |
| src/mlrose\_ky/fitness/queens.py                                 |       36 |        4 |     89% |72, 95, 145, 148 |
| src/mlrose\_ky/fitness/six\_peaks.py                             |       22 |        2 |     91% |    66, 87 |
| src/mlrose\_ky/fitness/travelling\_sales.py                      |       54 |        8 |     85% |57, 79, 81, 83, 162, 164, 166, 168 |
| src/mlrose\_ky/generators/\_\_init\_\_.py                        |        9 |        0 |    100% |           |
| src/mlrose\_ky/generators/continuous\_peaks\_generator.py        |       14 |        0 |    100% |           |
| src/mlrose\_ky/generators/flip\_flop\_generator.py               |        9 |        0 |    100% |           |
| src/mlrose\_ky/generators/four\_peaks\_generator.py              |       12 |        0 |    100% |           |
| src/mlrose\_ky/generators/knapsack\_generator.py                 |       23 |        0 |    100% |           |
| src/mlrose\_ky/generators/max\_k\_color\_generator.py            |       37 |        4 |     89% |     93-98 |
| src/mlrose\_ky/generators/one\_max\_generator.py                 |       11 |        0 |    100% |           |
| src/mlrose\_ky/generators/queens\_generator.py                   |       13 |        0 |    100% |           |
| src/mlrose\_ky/generators/six\_peaks\_generator.py               |       12 |        0 |    100% |           |
| src/mlrose\_ky/generators/tsp\_generator.py                      |       45 |        7 |     84% |64-69, 106 |
| src/mlrose\_ky/gridsearch/\_\_init\_\_.py                        |        1 |        0 |    100% |           |
| src/mlrose\_ky/gridsearch/grid\_search\_mixin.py                 |       33 |        2 |     94% |   129-130 |
| src/mlrose\_ky/neural/\_\_init\_\_.py                            |        7 |        0 |    100% |           |
| src/mlrose\_ky/neural/\_nn\_base.py                              |       61 |        3 |     95% |46, 64, 140 |
| src/mlrose\_ky/neural/\_nn\_core.py                              |      123 |       15 |     88% |78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 154, 157, 184, 263 |
| src/mlrose\_ky/neural/activation/\_\_init\_\_.py                 |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/identity.py                     |        7 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/leaky\_relu.py                  |       12 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/relu.py                         |       11 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/sigmoid.py                      |       10 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/softmax.py                      |       11 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/tanh.py                         |       10 |        0 |    100% |           |
| src/mlrose\_ky/neural/fitness/\_\_init\_\_.py                    |        1 |        0 |    100% |           |
| src/mlrose\_ky/neural/fitness/network\_weights.py                |       81 |        8 |     90% |54, 66, 69, 72, 78, 81, 84, 87 |
| src/mlrose\_ky/neural/linear\_regression.py                      |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/logistic\_regression.py                    |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/neural\_network.py                         |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/nn\_classifier.py                          |       75 |       41 |     45% |92-94, 110-117, 135, 155-212, 228-242, 258-259 |
| src/mlrose\_ky/neural/utils/\_\_init\_\_.py                      |        1 |        0 |    100% |           |
| src/mlrose\_ky/neural/utils/weights.py                           |       52 |        0 |    100% |           |
| src/mlrose\_ky/opt\_probs/\_\_init\_\_.py                        |        7 |        0 |    100% |           |
| src/mlrose\_ky/opt\_probs/\_opt\_prob.py                         |       69 |        5 |     93% |54, 56-59, 110 |
| src/mlrose\_ky/opt\_probs/continuous\_opt.py                     |       88 |       10 |     89% |54, 60, 62, 64, 114, 163, 203, 206, 214, 250 |
| src/mlrose\_ky/opt\_probs/discrete\_opt.py                       |      192 |       21 |     89% |81, 87, 89-92, 137-140, 162-164, 177-178, 246, 265, 289, 330, 362, 365, 392 |
| src/mlrose\_ky/opt\_probs/flip\_flop\_opt.py                     |       37 |        5 |     86% |63, 66, 103-106 |
| src/mlrose\_ky/opt\_probs/knapsack\_opt.py                       |       23 |        5 |     78% | 75, 80-83 |
| src/mlrose\_ky/opt\_probs/max\_k\_color\_opt.py                  |       50 |        3 |     94% |77, 84, 109 |
| src/mlrose\_ky/opt\_probs/queens\_opt.py                         |       25 |        2 |     92% |    61, 65 |
| src/mlrose\_ky/opt\_probs/tsp\_opt.py                            |       79 |        9 |     89% |53, 66-68, 77, 150, 192, 194-197 |
| src/mlrose\_ky/runners/\_\_init\_\_.py                           |        8 |        0 |    100% |           |
| src/mlrose\_ky/runners/\_nn\_runner\_base.py                     |      127 |       18 |     86% |169-171, 198-199, 206-207, 210-212, 282, 287, 300-301, 322-325, 404 |
| src/mlrose\_ky/runners/\_runner\_base.py                         |      273 |       51 |     81% |138, 157, 161, 170, 239-241, 254-257, 321, 350-358, 373-377, 409-415, 419-439, 478, 482, 542, 591, 674-675 |
| src/mlrose\_ky/runners/ga\_runner.py                             |       15 |        0 |    100% |           |
| src/mlrose\_ky/runners/mimic\_runner.py                          |       21 |        0 |    100% |           |
| src/mlrose\_ky/runners/nngs\_runner.py                           |       22 |        7 |     68% |128, 180-187 |
| src/mlrose\_ky/runners/rhc\_runner.py                            |       12 |        0 |    100% |           |
| src/mlrose\_ky/runners/sa\_runner.py                             |       20 |        1 |     95% |       106 |
| src/mlrose\_ky/runners/skmlp\_runner.py                          |      107 |       36 |     66% |133-140, 156, 176-188, 237-244, 249-263, 341, 407-412 |
| src/mlrose\_ky/runners/utils.py                                  |       14 |        2 |     86% |     37-38 |
| src/mlrose\_ky/samples/\_\_init\_\_.py                           |        1 |        0 |    100% |           |
| src/mlrose\_ky/samples/synthetic\_data.py                        |      123 |       28 |     77% |80-87, 171-190, 246, 260, 265-267, 276 |
| tests/\_\_init\_\_.py                                            |        0 |        0 |    100% |           |
| tests/globals.py                                                 |       10 |        0 |    100% |           |
| tests/test\_algorithms.py                                        |      108 |        0 |    100% |           |
| tests/test\_decay.py                                             |       32 |        0 |    100% |           |
| tests/test\_decorators.py                                        |       16 |        3 |     81% |15, 27, 37 |
| tests/test\_fitness.py                                           |       90 |        0 |    100% |           |
| tests/test\_generators/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| tests/test\_generators/test\_continuous\_peaks\_generator.py     |       52 |        0 |    100% |           |
| tests/test\_generators/test\_discrete\_four\_peaks\_generator.py |       50 |        0 |    100% |           |
| tests/test\_generators/test\_discrete\_six\_peaks\_generator.py  |       51 |        0 |    100% |           |
| tests/test\_generators/test\_flip\_flop\_generator.py            |       31 |        0 |    100% |           |
| tests/test\_generators/test\_knapsack\_generator.py              |       52 |        0 |    100% |           |
| tests/test\_generators/test\_max\_k\_color\_generator.py         |       86 |        0 |    100% |           |
| tests/test\_generators/test\_one\_max\_generator.py              |       35 |        0 |    100% |           |
| tests/test\_generators/test\_queens\_generator.py                |       39 |        0 |    100% |           |
| tests/test\_generators/test\_tsp\_generator.py                   |       50 |        0 |    100% |           |
| tests/test\_gridsearch.py                                        |      108 |        3 |     97% |52, 91, 157 |
| tests/test\_neural/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| tests/test\_neural/test\_linear\_regression.py                   |       57 |        0 |    100% |           |
| tests/test\_neural/test\_logistic\_regression.py                 |       64 |        0 |    100% |           |
| tests/test\_neural/test\_neural\_activation.py                   |       44 |        0 |    100% |           |
| tests/test\_neural/test\_neural\_fitness.py                      |      133 |        1 |     99% |       158 |
| tests/test\_neural/test\_neural\_network.py                      |       81 |        0 |    100% |           |
| tests/test\_neural/test\_neural\_utils.py                        |      135 |        0 |    100% |           |
| tests/test\_neural/test\_nn\_base.py                             |       53 |        0 |    100% |           |
| tests/test\_neural/test\_nn\_network\_weights.py                 |       59 |        0 |    100% |           |
| tests/test\_opt\_probs/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| tests/test\_opt\_probs/test\_continous\_opt.py                   |       93 |        0 |    100% |           |
| tests/test\_opt\_probs/test\_discrete\_opt.py                    |      100 |        0 |    100% |           |
| tests/test\_opt\_probs/test\_flip\_flop\_opt.py                  |       77 |        3 |     96% |66, 95, 102 |
| tests/test\_opt\_probs/test\_knapsack\_opt.py                    |       40 |        3 |     92% | 54-55, 60 |
| tests/test\_opt\_probs/test\_max\_k\_color\_opt.py               |       66 |        0 |    100% |           |
| tests/test\_opt\_probs/test\_opt\_prob.py                        |       81 |        0 |    100% |           |
| tests/test\_opt\_probs/test\_queens\_opt.py                      |       62 |        0 |    100% |           |
| tests/test\_opt\_probs/test\_tsp\_opt.py                         |       69 |        0 |    100% |           |
| tests/test\_runners/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| tests/test\_runners/test\_ga\_runner.py                          |       68 |        4 |     94% |74-75, 86-87 |
| tests/test\_runners/test\_mimic\_runner.py                       |       61 |        4 |     93% |71-72, 81-82 |
| tests/test\_runners/test\_nn\_runner\_base.py                    |       87 |        0 |    100% |           |
| tests/test\_runners/test\_nngs\_runner.py                        |       50 |        0 |    100% |           |
| tests/test\_runners/test\_rhc\_runner.py                         |       42 |        0 |    100% |           |
| tests/test\_runners/test\_runner\_base.py                        |       88 |        1 |     99% |        24 |
| tests/test\_runners/test\_runner\_utils.py                       |       47 |        0 |    100% |           |
| tests/test\_runners/test\_sa\_runner.py                          |       47 |        0 |    100% |           |
| tests/test\_runners/test\_skmlp\_runner.py                       |       53 |        0 |    100% |           |
| tests/test\_samples.py                                           |       40 |        0 |    100% |           |
|                                                        **TOTAL** | **5516** |  **497** | **91%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/knakamura13/mlrose-ky/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/knakamura13/mlrose-ky/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/knakamura13/mlrose-ky/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/knakamura13/mlrose-ky/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fknakamura13%2Fmlrose-ky%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/knakamura13/mlrose-ky/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.