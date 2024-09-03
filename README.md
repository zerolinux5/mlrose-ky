# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/knakamura13/mlrose-ky/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                             |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/mlrose\_ky/\_\_init\_\_.py                                   |       19 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/\_\_init\_\_.py                        |        9 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/crossovers/\_\_init\_\_.py             |        3 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/crossovers/\_crossover\_base.py        |       10 |        1 |     90% |        71 |
| src/mlrose\_ky/algorithms/crossovers/one\_point\_crossover.py    |        9 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/crossovers/tsp\_crossover.py           |       48 |       31 |     35% |86, 111-149 |
| src/mlrose\_ky/algorithms/crossovers/uniform\_crossover.py       |       10 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/decay/\_\_init\_\_.py                  |        4 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/decay/arithmetic\_decay.py             |       27 |       13 |     52% |50, 52, 54, 57, 60, 63-65, 100-112 |
| src/mlrose\_ky/algorithms/decay/custom\_decay.py                 |       21 |       10 |     52% |37, 40, 43-45, 79-90 |
| src/mlrose\_ky/algorithms/decay/exponential\_decay.py            |       28 |       13 |     54% |52, 54, 56, 59, 62, 65-67, 105-117 |
| src/mlrose\_ky/algorithms/decay/geometric\_decay.py              |       27 |       13 |     52% |50, 52, 54, 57, 60, 63-65, 103-115 |
| src/mlrose\_ky/algorithms/ga.py                                  |       87 |       22 |     75% |29, 48, 79-91, 208-212, 220-221, 252-253, 287-288 |
| src/mlrose\_ky/algorithms/gd.py                                  |       42 |        7 |     83% |72, 81, 110, 114-115, 126, 135 |
| src/mlrose\_ky/algorithms/hc.py                                  |       49 |       11 |     78% |97-99, 112, 116-128, 142-143, 147, 152 |
| src/mlrose\_ky/algorithms/mimic.py                               |       44 |        1 |     98% |        90 |
| src/mlrose\_ky/algorithms/mutators/\_\_init\_\_.py               |        4 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/mutators/\_mutator\_base.py            |       11 |        1 |     91% |        63 |
| src/mlrose\_ky/algorithms/mutators/discrete\_mutator.py          |       17 |       11 |     35% |34-35, 56-67 |
| src/mlrose\_ky/algorithms/mutators/gene\_swap\_mutator.py        |       11 |        0 |    100% |           |
| src/mlrose\_ky/algorithms/mutators/single\_gene\_mutator.py      |       14 |        1 |     93% |        60 |
| src/mlrose\_ky/algorithms/mutators/single\_shift\_mutator.py     |       14 |        8 |     43% |34-35, 55-61 |
| src/mlrose\_ky/algorithms/rhc.py                                 |       56 |       13 |     77% |105-107, 135-138, 142-155, 163, 167, 172 |
| src/mlrose\_ky/algorithms/sa.py                                  |       44 |        5 |     89% |94, 111, 135-136, 149 |
| src/mlrose\_ky/decorators/\_\_init\_\_.py                        |        1 |        0 |    100% |           |
| src/mlrose\_ky/decorators/short\_name\_decorator.py              |       12 |        0 |    100% |           |
| src/mlrose\_ky/fitness/\_\_init\_\_.py                           |       10 |        0 |    100% |           |
| src/mlrose\_ky/fitness/\_discrete\_peaks\_base.py                |       14 |        2 |     86% |    38, 66 |
| src/mlrose\_ky/fitness/continuous\_peaks.py                      |       33 |        3 |     91% |45, 105, 124 |
| src/mlrose\_ky/fitness/custom\_fitness.py                        |       15 |        3 |     80% |60, 85, 97 |
| src/mlrose\_ky/fitness/flip\_flop.py                             |       20 |        2 |     90% |    52, 78 |
| src/mlrose\_ky/fitness/four\_peaks.py                            |       20 |        2 |     90% |    65, 86 |
| src/mlrose\_ky/fitness/knapsack.py                               |       30 |        6 |     80% |72, 74, 76, 78, 104, 106 |
| src/mlrose\_ky/fitness/max\_k\_color.py                          |       18 |        1 |     94% |        92 |
| src/mlrose\_ky/fitness/one\_max.py                               |       11 |        1 |     91% |        53 |
| src/mlrose\_ky/fitness/queens.py                                 |       36 |        4 |     89% |72, 95, 145, 148 |
| src/mlrose\_ky/fitness/six\_peaks.py                             |       23 |        2 |     91% |    67, 88 |
| src/mlrose\_ky/fitness/travelling\_salesperson.py                |       54 |        8 |     85% |59, 82, 84, 86, 111, 113, 115, 117 |
| src/mlrose\_ky/generators/\_\_init\_\_.py                        |        9 |        0 |    100% |           |
| src/mlrose\_ky/generators/continuous\_peaks\_generator.py        |       15 |        0 |    100% |           |
| src/mlrose\_ky/generators/flip\_flop\_generator.py               |       10 |        0 |    100% |           |
| src/mlrose\_ky/generators/four\_peaks\_generator.py              |       13 |        0 |    100% |           |
| src/mlrose\_ky/generators/knapsack\_generator.py                 |       24 |        0 |    100% |           |
| src/mlrose\_ky/generators/max\_k\_color\_generator.py            |       38 |        4 |     89% |     93-98 |
| src/mlrose\_ky/generators/one\_max\_generator.py                 |       13 |        0 |    100% |           |
| src/mlrose\_ky/generators/queens\_generator.py                   |       14 |        0 |    100% |           |
| src/mlrose\_ky/generators/six\_peaks\_generator.py               |       13 |        0 |    100% |           |
| src/mlrose\_ky/generators/tsp\_generator.py                      |       45 |        7 |     84% |63-68, 104 |
| src/mlrose\_ky/gridsearch/\_\_init\_\_.py                        |        1 |        0 |    100% |           |
| src/mlrose\_ky/gridsearch/grid\_search\_mixin.py                 |       33 |        2 |     94% |   129-130 |
| src/mlrose\_ky/neural/\_\_init\_\_.py                            |        7 |        0 |    100% |           |
| src/mlrose\_ky/neural/\_nn\_base.py                              |       64 |        3 |     95% |39, 56, 82 |
| src/mlrose\_ky/neural/activation/\_\_init\_\_.py                 |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/identity.py                     |       10 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/leaky\_relu.py                  |       12 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/relu.py                         |       11 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/sigmoid.py                      |       10 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/softmax.py                      |       11 |        0 |    100% |           |
| src/mlrose\_ky/neural/activation/tanh.py                         |       10 |        0 |    100% |           |
| src/mlrose\_ky/neural/fitness/\_\_init\_\_.py                    |        1 |        0 |    100% |           |
| src/mlrose\_ky/neural/fitness/network\_weights.py                |       88 |        8 |     91% |45, 60, 64, 67, 73, 76, 79, 82 |
| src/mlrose\_ky/neural/linear\_regression.py                      |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/logistic\_regression.py                    |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/neural\_network.py                         |        6 |        0 |    100% |           |
| src/mlrose\_ky/neural/nn\_classifier.py                          |       74 |       41 |     45% |63-65, 68-73, 76, 80-134, 137-150, 153-154 |
| src/mlrose\_ky/neural/nn\_core.py                                |      134 |       20 |     85% |51, 81, 84, 87, 90, 93, 96, 99, 102, 104-107, 110, 113, 119, 160, 164, 193, 208, 293 |
| src/mlrose\_ky/neural/utils/\_\_init\_\_.py                      |        1 |        0 |    100% |           |
| src/mlrose\_ky/neural/utils/weights.py                           |       54 |        0 |    100% |           |
| src/mlrose\_ky/opt\_probs/\_\_init\_\_.py                        |        7 |        0 |    100% |           |
| src/mlrose\_ky/opt\_probs/continuous\_opt.py                     |       94 |       15 |     84% |41, 49, 52, 55, 101, 149, 151-154, 190, 193, 201-204, 235 |
| src/mlrose\_ky/opt\_probs/discrete\_opt.py                       |      201 |       27 |     87% |43, 51, 53-56, 108-113, 134-136, 150-151, 242, 261, 291, 335, 337-340, 373, 376, 408, 410-413 |
| src/mlrose\_ky/opt\_probs/flip\_flop\_opt.py                     |       38 |        5 |     87% |18, 21, 52-55 |
| src/mlrose\_ky/opt\_probs/knapsack\_opt.py                       |       22 |        5 |     77% | 27, 32-35 |
| src/mlrose\_ky/opt\_probs/max\_k\_color\_opt.py                  |       49 |        3 |     94% |20, 26, 49 |
| src/mlrose\_ky/opt\_probs/opt\_prob.py                           |       79 |        5 |     94% |26, 28-31, 96 |
| src/mlrose\_ky/opt\_probs/queens\_opt.py                         |       24 |        2 |     92% |    18, 21 |
| src/mlrose\_ky/opt\_probs/tsp\_opt.py                            |       80 |        9 |     89% |51, 59-60, 65, 140, 182, 184-187 |
| src/mlrose\_ky/runners/\_\_init\_\_.py                           |        8 |        0 |    100% |           |
| src/mlrose\_ky/runners/\_nn\_runner\_base.py                     |      150 |       21 |     86% |163, 179-181, 209-210, 217-218, 221-223, 302, 312, 328-330, 347, 352-355, 439 |
| src/mlrose\_ky/runners/\_runner\_base.py                         |      275 |       51 |     81% |101, 145, 149, 159, 302-304, 318-321, 389, 416-422, 437-440, 484-488, 499-519, 558, 561, 620, 668, 742-743 |
| src/mlrose\_ky/runners/ga\_runner.py                             |       14 |        0 |    100% |           |
| src/mlrose\_ky/runners/mimic\_runner.py                          |       20 |        0 |    100% |           |
| src/mlrose\_ky/runners/nngs\_runner.py                           |       21 |        7 |     67% |67, 102-109 |
| src/mlrose\_ky/runners/rhc\_runner.py                            |       11 |        0 |    100% |           |
| src/mlrose\_ky/runners/sa\_runner.py                             |       19 |        1 |     95% |        59 |
| src/mlrose\_ky/runners/skmlp\_runner.py                          |      104 |       36 |     65% |61-65, 68, 71-79, 82-87, 90-104, 144, 191-196 |
| src/mlrose\_ky/runners/utils.py                                  |       14 |        2 |     86% |     37-38 |
| src/mlrose\_ky/samples/\_\_init\_\_.py                           |        1 |        0 |    100% |           |
| src/mlrose\_ky/samples/synthetic\_data.py                        |      122 |       28 |     77% |79-86, 169-189, 243, 257, 262-264, 273 |
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
| tests/test\_generators/test\_tsp\_generator.py                   |       49 |        0 |    100% |           |
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
| tests/test\_opt\_probs/test\_flip\_flop\_opt.py                  |       76 |        3 |     96% |66, 94, 101 |
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
|                                                        **TOTAL** | **5602** |  **519** | **91%** |           |


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