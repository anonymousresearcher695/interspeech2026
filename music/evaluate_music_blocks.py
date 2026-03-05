#!/usr/bin/env python3
"""
Music Block Evaluation System
Ground Truth vs Predicted
"""
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from config import OUTPUT_DIR


# Ground Truth — {broadcast}: {date} [intervals]}}
GROUND_TRUTH_MUSIC = {
    'SBS': {
        '20241120': [ # 10
            {'start': 126.40, 'end': 371.27, 'type': 'MUSIC'},
            {'start': 854.6, 'end': 1109.03, 'type': 'MUSIC'},
            {'start': 1302.02, 'end': 1416.79, 'type': 'MUSIC'},
            {'start': 1548.64, 'end': 1663.60, 'type': 'MUSIC'},
            {'start': 1927.28, 'end': 2092.25, 'type': 'MUSIC'},
            {'start': 2248.83, 'end': 2358.93, 'type': 'MUSIC'},
            {'start': 2407.47, 'end': 2590.82, 'type': 'MUSIC'},
            {'start': 2630.28, 'end': 2790.87, 'type': 'MUSIC'},
            {'start': 2833.94, 'end': 3039.68, 'type': 'MUSIC'},
            {'start': 3301.5, 'end': 3445.93, 'type': 'MUSIC'},
        ],
        '20241121': [ # 7
            {'start': 128.05, 'end': 350.53, 'type': 'MUSIC'},
            {'start': 730.34, 'end': 923.95, 'type': 'MUSIC'},
            {'start': 1043.59, 'end': 1265.42, 'type': 'MUSIC'},
            {'start': 1430.42, 'end': 1663.90, 'type': 'MUSIC'},
            {'start': 2428.82, 'end': 2613.27, 'type': 'MUSIC'},
            {'start': 2914.75, 'end': 3063.00, 'type': 'MUSIC'},
            {'start': 3285.00, 'end': 3446.00, 'type': 'MUSIC'}
        ],
        '20241122': [ # 3
            {'start': 158.78, 'end': 379.68, 'type': 'MUSIC'},
            {'start': 961.02, 'end': 1248.77, 'type': 'MUSIC'},
            {'start': 1599.49, 'end': 1662.52, 'type': 'MUSIC'},
        ],
        '20241123': [ # 9
            {'start': 143.00, 'end': 335.99, 'type': 'MUSIC'},
            {'start': 645.09, 'end': 791.20, 'type': 'MUSIC'},
            {'start': 943.88, 'end': 1381.02, 'type': 'MUSIC'},
            {'start': 1538.09, 'end': 1662.94, 'type': 'MUSIC'},
            {'start': 1925.15, 'end': 2075.98, 'type': 'MUSIC'},
            {'start': 2222.00, 'end': 2374.00, 'type': 'MUSIC'},
            {'start': 2431.82, 'end': 2609.62, 'type': 'MUSIC'},
            {'start': 2717.10, 'end': 2983.88, 'type': 'MUSIC'},
            {'start': 3269.47, 'end': 3445.14, 'type': 'MUSIC'},
        ],
        '20241124': [ # 9
            {'start': 144.34, 'end': 342.95,'type': 'MUSIC'},
            {'start': 610.00, 'end': 732.44,'type': 'MUSIC'},
            {'start': 888.38, 'end': 1423.54,'type': 'MUSIC'},
            {'start': 1548.77, 'end': 1663.22,'type': 'MUSIC'},
            {'start': 1925.52, 'end': 2115.34,'type': 'MUSIC'},
            {'start': 2228.81, 'end': 2406.25,'type': 'MUSIC'},
            {'start': 2633.05, 'end': 2740.20,'type': 'MUSIC'},
            {'start': 2792.44, 'end': 2992.32,'type': 'MUSIC'},
            {'start': 3255.89, 'end': 3444.63,'type': 'MUSIC'},
        ],
        '20241125': [ # 8
            {'start': 134.71, 'end': 374.40, 'type': 'MUSIC'},
            {'start': 865.85, 'end': 967.37, 'type': 'MUSIC'},
            {'start': 1093.34, 'end': 1337.45, 'type': 'MUSIC'},
            {'start': 1466.00, 'end': 1662.88, 'type': 'MUSIC'},
            {'start': 1925.07, 'end': 2050.25, 'type': 'MUSIC'},
            {'start': 2327.07, 'end': 2442.63, 'type': 'MUSIC'},
            {'start': 2594.80, 'end': 2766.20, 'type': 'MUSIC'},
            {'start': 3142.23, 'end': 3444.58, 'type': 'MUSIC'},
        ],
        '20241126': [ # 5
            {'start': 139.31, 'end': 589.49, 'type': 'MUSIC'},
            {'start': 1174.53, 'end': 1426.16, 'type': 'MUSIC'},
            {'start': 1563.01, 'end': 1665.57, 'type': 'MUSIC'},
            {'start': 2338.27, 'end': 2534.01, 'type': 'MUSIC'},
            {'start': 3261.98, 'end': 3447.47, 'type': 'MUSIC'},
        ],
        '20241127': [ # 4
            {'start': 130.17, 'end': 358.74, 'type': 'MUSIC'},
            {'start': 1276.59, 'end': 1471.87, 'type': 'MUSIC'},
            {'start': 2429.66, 'end': 2648.69, 'type': 'MUSIC'},
            {'start': 3375.48, 'end': 3452.11, 'type': 'MUSIC'},
        ],
        '20241128': [ # 6
            {'start': 141.19,'end': 268.56, 'type': 'MUSIC'},
            {'start': 854.03,'end': 1044.18, 'type': 'MUSIC'},
            {'start': 1211.19,'end': 1408.69, 'type': 'MUSIC'},
            {'start': 1509.55,'end': 1625.23, 'type': 'MUSIC'},
            {'start': 2416.98,'end': 2602.96, 'type': 'MUSIC'},
            {'start': 3255.84,'end': 3411.71, 'type': 'MUSIC'},
        ],
        '20241129': [ # 3
            {'start': 142.38, 'end': 311.13, 'type': 'MUSIC'},
            {'start': 885.78, 'end': 1058.29, 'type': 'MUSIC'},
            {'start': 1300.47, 'end': 1665.31, 'type': 'MUSIC'},
        ],
    },
    'MBC': {
        '20241124': [ # 17
            {'start': 143.9, 'end': 353.14, 'type': 'MUSIC'},
            {'start': 490.34, 'end': 644.42, 'type': 'MUSIC'},
            {'start': 728.09, 'end': 1163.95, 'type': 'MUSIC'},
            {'start': 1220.42, 'end': 1488.9, 'type': 'MUSIC'},
            {'start': 1500.0, 'end': 1717.55, 'type': 'MUSIC'},
            {'start': 1877.85, 'end': 2324.36, 'type': 'MUSIC'},
            {'start': 2538.07, 'end': 2929.83, 'type': 'MUSIC'},
            {'start': 3242.74, 'end': 3367.09, 'type': 'MUSIC'},
            {'start': 3706.96, 'end': 3844.85, 'type': 'MUSIC'},
            {'start': 3941.32, 'end': 4219.36, 'type': 'MUSIC'},
            {'start': 4395.98, 'end': 4897.4, 'type': 'MUSIC'},
            {'start': 4945.27, 'end': 5214.74, 'type': 'MUSIC'},
            {'start': 5231.91, 'end': 5319.79, 'type': 'MUSIC'},
            {'start': 5477.85, 'end': 5704.13, 'type': 'MUSIC'},
            {'start': 5864.95, 'end': 6275.31, 'type': 'MUSIC'},
            {'start': 6414.24, 'end': 6636.94, 'type': 'MUSIC'},
            {'start': 6855.06, 'end': 6966.87, 'type': 'MUSIC'}
        ],
        '20241125': [ # 13
            {'start': 149.52, 'end': 398.54, 'type': 'MUSIC'},
            {'start': 575.60, 'end': 838.50, 'type': 'MUSIC'},
            {'start': 1104.46, 'end': 1578.63, 'type': 'MUSIC'},
            {'start': 2145.92, 'end': 2400.13, 'type': 'MUSIC'},
            {'start': 2640.57, 'end': 2873.58, 'type': 'MUSIC'},
            {'start': 3118.81, 'end': 3365.20, 'type': 'MUSIC'},
            {'start': 3612.60, 'end': 3829.62, 'type': 'MUSIC'},
            {'start': 4709.97, 'end': 4911.28, 'type': 'MUSIC'},
            {'start': 5162.34, 'end': 5319.21, 'type': 'MUSIC'},
            {'start': 5476.44, 'end': 5651.18, 'type': 'MUSIC'},
            {'start': 5749.23, 'end': 6183.00, 'type': 'MUSIC'},
            {'start': 6333.10, 'end': 6541.94, 'type': 'MUSIC'},
            {'start': 6758.05, 'end': 6964.93, 'type': 'MUSIC'}
        ],
        '20241126': [ # 13
            {'start': 148.30, 'end': 408.34, 'type': 'MUSIC'},
            {'start': 600.73, 'end': 798.57, 'type': 'MUSIC'},
            {'start': 885.64, 'end': 1350.62, 'type': 'MUSIC'},
            {'start': 1491.51, 'end': 1720.27, 'type': 'MUSIC'},
            {'start': 2145.07, 'end': 2274.08, 'type': 'MUSIC'},
            {'start': 2475.97, 'end': 2892.07, 'type': 'MUSIC'},
            {'start': 3154.40, 'end': 3367.61, 'type': 'MUSIC'},
            {'start': 3614.22, 'end': 3793.99, 'type': 'MUSIC'},
            {'start': 4567.15, 'end': 4758.16, 'type': 'MUSIC'},
            {'start': 4957.01, 'end': 5320.70, 'type': 'MUSIC'},
            {'start': 5478.29, 'end': 6007.19, 'type': 'MUSIC'},
            {'start': 6085.17, 'end': 6458.70, 'type': 'MUSIC'},
            {'start': 6776.76, 'end': 6966.68, 'type': 'MUSIC'}
        ],
        '20260101': [ # 18
            {'start': 169.2, 'end': 359.06, 'type': 'MUSIC'},
            {'start': 607.65, 'end': 848.04, 'type': 'MUSIC'},
            {'start': 871.53, 'end': 1229.31, 'type': 'MUSIC'},
            {'start': 1261.76, 'end': 1669.83, 'type': 'MUSIC'},
            {'start': 1878.00, 'end': 2032.52, 'type': 'MUSIC'},
            {'start': 2059.29, 'end': 2498.4, 'type': 'MUSIC'},
            {'start': 2540.64, 'end': 3012.57, 'type': 'MUSIC'},
            {'start': 3211.47, 'end': 3366.8, 'type': 'MUSIC'},
            {'start': 3613.49, 'end': 3788.5, 'type': 'MUSIC'},
            {'start': 3921.89, 'end': 4094.17, 'type': 'MUSIC'},
            {'start': 4125.85, 'end': 4582.81, 'type': 'MUSIC'},
            {'start': 4606.9, 'end': 4949.29, 'type': 'MUSIC'},
            {'start': 5013.81, 'end': 5196.53, 'type': 'MUSIC'},
            {'start': 5256.71, 'end': 5319.99, 'type': 'MUSIC'},
            {'start': 5477.81, 'end': 5656.0, 'type': 'MUSIC'},
            {'start': 5680.2, 'end': 6094.65, 'type': 'MUSIC'},
            {'start': 6133.85, 'end': 6480.0, 'type': 'MUSIC'},
            {'start': 6711.3, 'end': 6966.85, 'type': 'MUSIC'},
        ],
        '20260102': [ # 18
            {'start': 195.89, 'end': 382.1, 'type': 'MUSIC'},
            {'start': 647.72, 'end': 787.89, 'type': 'MUSIC'},
            {'start': 839.32, 'end': 1181.19, 'type': 'MUSIC'},
            {'start': 1234.33, 'end': 1560.29, 'type': 'MUSIC'},
            {'start': 1596.39, 'end': 1721.21, 'type': 'MUSIC'},
            {'start': 1906.72, 'end': 2097.78, 'type': 'MUSIC'},
            {'start': 2131.87, 'end': 2533.96, 'type': 'MUSIC'},
            {'start': 2586.28, 'end': 2912.18, 'type': 'MUSIC'},
            {'start': 3166.58, 'end': 3369.75, 'type': 'MUSIC'},
            {'start': 3615.48, 'end': 3858.23, 'type': 'MUSIC'},
            {'start': 4016.99, 'end': 4271.63, 'type': 'MUSIC'},
            {'start': 4313.15, 'end': 4672.1, 'type': 'MUSIC'},
            {'start': 4746.31, 'end': 5143.52, 'type': 'MUSIC'},
            {'start': 5213.78, 'end': 5322.19, 'type': 'MUSIC'},
            {'start': 5480.1, 'end': 5703.95, 'type': 'MUSIC'},
            {'start': 5925.67, 'end': 6077.97, 'type': 'MUSIC'},
            {'start': 6141.94, 'end': 6475.64, 'type': 'MUSIC'},
            {'start': 6679.54, 'end': 6968.69, 'type': 'MUSIC'},
        ],
        '20260103': [ # 18
            {'start': 198.77, 'end': 508.12, 'type': 'MUSIC'},
            {'start': 753.87, 'end': 941.69, 'type': 'MUSIC'},
            {'start': 989.48, 'end': 1362.86, 'type': 'MUSIC'},
            {'start': 1392.15, 'end': 1722.23, 'type': 'MUSIC'},
            {'start': 1929.97, 'end': 2163.86, 'type': 'MUSIC'},
            {'start': 2189.78, 'end': 2576.87, 'type': 'MUSIC'},
            {'start': 2598.50, 'end': 2849.40, 'type': 'MUSIC'},
            {'start': 2901.34, 'end': 3108.02, 'type': 'MUSIC'},
            {'start': 3300.66, 'end': 3368.55, 'type': 'MUSIC'},
            {'start': 3613.00, 'end': 3888.67, 'type': 'MUSIC'},
            {'start': 4043.50, 'end': 4216.86, 'type': 'MUSIC'},
            {'start': 4304.97, 'end': 4706.28, 'type': 'MUSIC'},
            {'start': 4757.54, 'end': 5151.97, 'type': 'MUSIC'},
            {'start': 5224.03, 'end': 5320.84, 'type': 'MUSIC'},
            {'start': 5478.00, 'end': 5702.13, 'type': 'MUSIC'},
            {'start': 5730.33, 'end': 6213.10, 'type': 'MUSIC'},
            {'start': 6292.42, 'end': 6556.74, 'type': 'MUSIC'},
            {'start': 6702.49, 'end': 6967.00, 'type': 'MUSIC'},
        ],
        '20260104': [ # 17
            {'start': 188.95, 'end': 373.14, 'type': 'MUSIC'},
            {'start': 654.50, 'end': 888.74, 'type': 'MUSIC'},
            {'start': 944.72, 'end': 1423.31, 'type': 'MUSIC'},
            {'start': 1477.90, 'end': 1696.00, 'type': 'MUSIC'},
            {'start': 1877.89, 'end': 2138.02, 'type': 'MUSIC'},
            {'start': 2246.77, 'end': 2772.38, 'type': 'MUSIC'},
            {'start': 2808.92, 'end': 3053.59, 'type': 'MUSIC'},
            {'start': 3244.81, 'end': 3365.67, 'type': 'MUSIC'},
            {'start': 3613.38, 'end': 3843.80, 'type': 'MUSIC'},
            {'start': 3966.77, 'end': 4196.50, 'type': 'MUSIC'},
            {'start': 4274.01, 'end': 4669.29, 'type': 'MUSIC'},
            {'start': 4763.92, 'end': 5063.92, 'type': 'MUSIC'},
            {'start': 5106.98, 'end': 5291.89, 'type': 'MUSIC'},
            {'start': 5476.21, 'end': 5750.25, 'type': 'MUSIC'},
            {'start': 5829.03, 'end': 6244.32, 'type': 'MUSIC'},
            {'start': 6276.00, 'end': 6528.21, 'type': 'MUSIC'},
            {'start': 6649.48, 'end': 6824.28, 'type': 'MUSIC'},
        ],
        '20260105': [ #13
            {'start': 144.58, 'end': 419.33, 'type': 'MUSIC'},
            {'start': 806.92, 'end': 1040.81, 'type': 'MUSIC'},
            {'start': 1226.33, 'end': 1608.40, 'type': 'MUSIC'},
            {'start': 2142.59, 'end': 2410.24, 'type': 'MUSIC'},
            {'start': 2644.84, 'end': 2856.03, 'type': 'MUSIC'},
            {'start': 3155.35, 'end': 3367.84, 'type': 'MUSIC'},
            {'start': 3612.92, 'end': 3843.22, 'type': 'MUSIC'},
            {'start': 4780.48, 'end': 4978.24, 'type': 'MUSIC'},
            {'start': 5267.83, 'end': 5323.62, 'type': 'MUSIC'},
            {'start': 5477.48, 'end': 5649.17, 'type': 'MUSIC'},
            {'start': 5681.04, 'end': 6088.48, 'type': 'MUSIC'},
            {'start': 6248.32, 'end': 6610.75, 'type': 'MUSIC'},
            {'start': 6790.32, 'end': 6864.10, 'type': 'MUSIC'},
        ],

        '20260106': [ # 10
            {'start': 152.28, 'end': 307.04, 'type': 'MUSIC'},
            {'start': 618.29, 'end': 762.88, 'type': 'MUSIC'},
            {'start': 933.82, 'end': 1469.88, 'type': 'MUSIC'},
            {'start': 1508.51, 'end': 1718.96, 'type': 'MUSIC'},
            {'start': 2155.52, 'end': 2653.73, 'type': 'MUSIC'},
            {'start': 2788.81, 'end': 2931.53, 'type': 'MUSIC'},
            {'start': 3115.99, 'end': 3366.02, 'type': 'MUSIC'},
            {'start': 3612.69, 'end': 3825.11, 'type': 'MUSIC'},
            {'start': 4434.23, 'end': 4658.72, 'type': 'MUSIC'},
            {'start': 4923.23, 'end': 5318.10, 'type': 'MUSIC'},
        ],

        '20260107': [ # 9
            {'start': 151.61, 'end': 468.11, 'type': 'MUSIC'},
            {'start': 878.79, 'end': 1131.40, 'type': 'MUSIC'},
            {'start': 1229.50, 'end': 1681.45, 'type': 'MUSIC'},
            {'start': 2130.40, 'end': 2464.73, 'type': 'MUSIC'},
            {'start': 2524.64, 'end': 2858.77, 'type': 'MUSIC'},
            {'start': 3195.92, 'end': 3238.04, 'type': 'MUSIC'},
            {'start': 3275.95, 'end': 3366.20, 'type': 'MUSIC'},
            {'start': 3613.08, 'end': 3828.12, 'type': 'MUSIC'},
            {'start': 5185.42, 'end': 5319.79, 'type': 'MUSIC'},
            {'start': 6861.42, 'end': 6966.88, 'type': 'MUSIC'},
        ],
        
    },
    'KBS': {
        '20260101': [ # 11
            {'start': 79.23, 'end': 254.54, 'type': 'MUSIC'},
            {'start': 1124.94, 'end': 1340.53, 'type': 'MUSIC'},
            {'start': 1571.71, 'end': 1719.27, 'type': 'MUSIC'},
            {'start': 1948.17, 'end': 2220.52, 'type': 'MUSIC'},
            {'start': 2819.00, 'end': 2975.14, 'type': 'MUSIC'},
            {'start': 3254.71, 'end': 3438.02, 'type': 'MUSIC'},
            {'start': 4223.56, 'end': 4449.55, 'type': 'MUSIC'},
            {'start': 4847.60, 'end': 5079.64, 'type': 'MUSIC'},
            {'start': 5786.44, 'end': 5996.57, 'type': 'MUSIC'},
            {'start': 6487.56, 'end': 6740.12, 'type': 'MUSIC'},
            {'start': 6889.64, 'end': 7038.69, 'type': 'MUSIC'}
        ],
        '20260102': [ # 10
            {'start': 82.30, 'end': 274.28, 'type': 'MUSIC'},
            {'start': 1151.88, 'end': 1366.99, 'type': 'MUSIC'},
            {'start': 1578.15, 'end': 1718.10, 'type': 'MUSIC'},
            {'start': 1939.40, 'end': 2130.00, 'type': 'MUSIC'},
            {'start': 2732.38, 'end': 2936.71, 'type': 'MUSIC'},
            {'start': 3297.59, 'end': 3438.13, 'type': 'MUSIC'},
            {'start': 4524.57, 'end': 4716.68, 'type': 'MUSIC'},
            {'start': 5157.24, 'end': 5317.62, 'type': 'MUSIC'},
            {'start': 6548.94, 'end': 6742.35, 'type': 'MUSIC'},
            {'start': 6879.70, 'end': 7040.00, 'type': 'MUSIC'}
        ],
        '20260108': [ # 9
            {'start': 92.52, 'end': 280.69, 'type': 'MUSIC'},
            {'start': 1233.67, 'end': 1459.52, 'type': 'MUSIC'},
            {'start': 1597.15, 'end': 1714.74, 'type': 'MUSIC'},
            {'start': 1945.50, 'end': 2175.72, 'type': 'MUSIC'},
            {'start': 2812.13, 'end': 2990.76, 'type': 'MUSIC'},
            {'start': 3270.88, 'end': 3434.38, 'type': 'MUSIC'},
            {'start': 4540.61, 'end': 4748.30, 'type': 'MUSIC'},
            {'start': 5193.97, 'end': 5314.11, 'type': 'MUSIC'},
            {'start': 6656.61, 'end': 6836.89, 'type': 'MUSIC'}
        ],
        '20260109': [ # 9
            {'start': 86.02, 'end': 303.03, 'type': 'MUSIC'},
            {'start': 1167.57, 'end': 1356.89, 'type': 'MUSIC'},
            {'start': 3310.69, 'end': 3435.79, 'type': 'MUSIC'},
            {'start': 4473.37, 'end': 4727.23, 'type': 'MUSIC'},
            {'start': 4921.19, 'end': 5125.08, 'type': 'MUSIC'}, 
            {'start': 5240.74, 'end': 5315.58, 'type': 'MUSIC'}, 
            {'start': 5540.34, 'end': 5736.48, 'type': 'MUSIC'},
            {'start': 6332.69, 'end': 6534.52, 'type': 'MUSIC'},
            {'start': 6740.00, 'end': 6916.39, 'type': 'MUSIC'},
        ],
        '20260110': [ #20
            {'start': 79.36, 'end': 297.19, 'type': 'MUSIC'},
            {'start': 721.19, 'end': 864.11, 'type': 'MUSIC'},
            {'start': 1054.65, 'end': 1299.49, 'type': 'MUSIC'},
            {'start': 1485.91, 'end': 1713.33, 'type': 'MUSIC'},
            {'start': 2058.45, 'end': 2254.65, 'type': 'MUSIC'},
            {'start': 2281.23, 'end': 2459.97, 'type': 'MUSIC'},   
            {'start': 2476.65, 'end': 2686.39, 'type': 'MUSIC'},
            {'start': 2711.02, 'end': 2881.46, 'type': 'MUSIC'}, 
            {'start': 2916.32, 'end': 3049.50, 'type': 'MUSIC'}, 
            {'start': 3300.61, 'end': 3433.25, 'type': 'MUSIC'},
            {'start': 4047.53, 'end': 4232.75, 'type': 'MUSIC'},
            {'start': 4257.06, 'end': 4398.10, 'type': 'MUSIC'},
            {'start': 4412.24, 'end': 4613.95, 'type': 'MUSIC'},
            {'start': 4642.52, 'end': 4816.98, 'type': 'MUSIC'},
            {'start': 4846.52, 'end': 5000.87, 'type': 'MUSIC'},
            {'start': 5164.80, 'end': 5313.34, 'type': 'MUSIC'},
            {'start': 5559.90, 'end': 5714.31, 'type': 'MUSIC'},
            {'start': 5918.07, 'end': 6430.88, 'type': 'MUSIC'},
            {'start': 6514.29, 'end': 6677.68, 'type': 'MUSIC'}, 
            {'start': 6843.35, 'end': 7034.40, 'type': 'MUSIC'},
        ],
        '20260111': [ # 13
            {'start': 78.83, 'end': 256.05, 'type': 'MUSIC'},
            {'start': 714.94, 'end': 927.31, 'type': 'MUSIC'}, 
            {'start': 1164.23, 'end': 1361.79, 'type': 'MUSIC'},
            {'start': 1550.88, 'end': 1717.59, 'type': 'MUSIC'}, 
            {'start': 2324.51, 'end': 2518.48, 'type': 'MUSIC'},
            {'start': 2993.41, 'end': 3210.51, 'type': 'MUSIC'}, 
            {'start': 4331.97, 'end': 4552.07, 'type': 'MUSIC'},
            {'start': 4937.39, 'end': 5083.79, 'type': 'MUSIC'},
            {'start': 5161.32, 'end': 5315.57, 'type': 'MUSIC'}, 
            {'start': 5572.11, 'end': 5709.57, 'type': 'MUSIC'},
            {'start': 5829.09, 'end': 6334.59, 'type': 'MUSIC'}, 
            {'start': 6440.84, 'end': 6656.45, 'type': 'MUSIC'},
            {'start': 6808.12, 'end': 7033.00, 'type': 'MUSIC'},
        ],
        '20260112': [ # 8
            {'start': 90.98, 'end': 256.05, 'type': 'MUSIC'},
            {'start': 1222.50, 'end': 1435.73, 'type': 'MUSIC'}, 
            {'start': 1540.53, 'end': 1718.94, 'type': 'MUSIC'},
            {'start': 1945.68, 'end': 2146.58, 'type': 'MUSIC'}, 
            {'start': 2611.80, 'end': 2817.97, 'type': 'MUSIC'},
            {'start': 3244.62, 'end': 3438.95, 'type': 'MUSIC'}, 
            {'start': 4715.08, 'end': 4860.13, 'type': 'MUSIC'}, 
            {'start': 6719.15, 'end': 6870.04, 'type': 'MUSIC'},
        ],
        '20260113': [ # 11
            {'start': 82.56, 'end': 290.27, 'type': 'MUSIC'},
            {'start': 1154.95, 'end': 1372.98, 'type': 'MUSIC'}, 
            {'start': 1532.99, 'end': 1717.30, 'type': 'MUSIC'}, 
            {'start': 1947.41, 'end': 2175.48, 'type': 'MUSIC'}, 
            {'start': 2716.11, 'end': 2901.77, 'type': 'MUSIC'},
            {'start': 3283.08, 'end': 3436.91, 'type': 'MUSIC'}, 
            {'start': 4336.53, 'end': 4587.51, 'type': 'MUSIC'},
            {'start': 5125.66, 'end': 5317.86, 'type': 'MUSIC'},
            {'start': 5834.30, 'end': 5984.63, 'type': 'MUSIC'}, 
            {'start': 6335.78, 'end': 6550.47, 'type': 'MUSIC'},
            {'start': 6823.51, 'end': 7037.29, 'type': 'MUSIC'},  
        ],
        '20260114': [ # 8
            {'start': 81.29, 'end': 296.64, 'type': 'MUSIC'},
            {'start': 1039.48, 'end': 1225.93, 'type': 'MUSIC'},
            {'start': 1551.00, 'end': 1717.36, 'type': 'MUSIC'},  
            {'start': 1940.72, 'end': 2185.91, 'type': 'MUSIC'},
            {'start': 2613.44, 'end': 2835.72, 'type': 'MUSIC'},
            {'start': 3289.66, 'end': 3435.99, 'type': 'MUSIC'}, 
            {'start': 4028.70, 'end': 4285.54, 'type': 'MUSIC'},
            {'start': 6573.47, 'end': 6796.53, 'type': 'MUSIC'}, 
        ],
        '20260115': [ # 8
            {'start': 87.25, 'end': 282.07, 'type': 'MUSIC'},
            {'start': 1122.87, 'end': 1330.92, 'type': 'MUSIC'},
            {'start': 1556.63, 'end': 1716.78, 'type': 'MUSIC'},  
            {'start': 1941.26, 'end': 2149.06, 'type': 'MUSIC'}, 
            {'start': 2957.57, 'end': 3179.86, 'type': 'MUSIC'},
            {'start': 4514.56, 'end': 4686.13, 'type': 'MUSIC'},
            {'start': 5151.68, 'end': 5316.82, 'type': 'MUSIC'},
            {'start': 6647.04, 'end': 6837.78, 'type': 'MUSIC'},
        ]
    }
}


class MusicEvaluator:
    def __init__(self, broadcaster, date_str):
        self.broadcaster = broadcaster.upper()
        self.date_str    = date_str

        bc_data = GROUND_TRUTH_MUSIC.get(self.broadcaster, {})
        self.gt_music = bc_data.get(date_str)

        if not self.gt_music:
            available = {
                bc: list(dates.keys())
                for bc, dates in GROUND_TRUTH_MUSIC.items()
            }
            print(f"\n❌ Error: No GT data for {self.broadcaster} / {date_str}")
            print(f"   Available:")
            for bc, dates in available.items():
                print(f"      {bc}: {', '.join(dates)}")
            sys.exit(1)

    def load_predicted_blocks(self, csv_file):
        df = pd.read_csv(csv_file)
        predicted = []
        if 'block_type' in df.columns:
            for _, row in df.iterrows():
                predicted.append({
                    'start': float(row['start']),
                    'end':   float(row['end']),
                    'type':  row['block_type'].strip()
                })
        elif 'duration' in df.columns and 'start' in df.columns:
            for _, row in df.iterrows():
                predicted.append({
                    'start': float(row['start']),
                    'end':   float(row['end']),
                    'type':  'MUSIC'
                })
        else:
            raise ValueError(f"Unknown CSV format. Columns: {df.columns.tolist()}")
        return predicted

    def calculate_iou(self, gt, pred):
        overlap = max(0, min(gt['end'], pred['end']) - max(gt['start'], pred['start']))
        if overlap == 0:
            return 0.0
        union = (gt['end'] - gt['start']) + (pred['end'] - pred['start']) - overlap
        return overlap / union if union > 0 else 0.0


    def evaluate_all(self, predicted_blocks):
        print(f"\n{'='*70}")
        print(f"🎵 Music Block Evaluation: [{self.broadcaster}] {self.date_str}")
        print(f"{'='*70}\n")

        pred_music     = [b for b in predicted_blocks if b['type'] == 'MUSIC']
        pred_non_music = [b for b in predicted_blocks if b['type'] != 'MUSIC']

        print(f"📊 Dataset Summary:")
        print(f"   GT MUSIC blocks:          {len(self.gt_music)}")
        print(f"   Predicted MUSIC blocks:   {len(pred_music)}")
        print(f"   Predicted Non-MUSIC blocks: {len(pred_non_music)}")

        print(f"\n{'='*70}")
        print("📍 [1/5] Music Detection Accuracy (IoU-based)")
        print(f"{'='*70}")
        detection_results = self.music_detection_accuracy(pred_music)

        print(f"\n{'='*70}")
        print("⏱️  [2/5] Temporal IoU Analysis")
        print(f"{'='*70}")
        temporal_iou = self.temporal_iou_analysis(pred_music)

        print(f"\n{'='*70}")
        print("🎯 [3/5] Boundary Precision & Recall")
        print(f"{'='*70}")
        boundary_metrics = self.boundary_evaluation(pred_music)

        print(f"\n{'='*70}")
        print("❌ [4/5] False Positive Analysis")
        print(f"{'='*70}")
        fp_analysis = self.false_positive_analysis(pred_music, predicted_blocks)

        print(f"\n{'='*70}")
        print("⚠️  [5/5] False Negative Analysis")
        print(f"{'='*70}")
        fn_analysis = self.false_negative_analysis(pred_music, predicted_blocks)

        self.print_summary(detection_results, temporal_iou, boundary_metrics,
                           fp_analysis, fn_analysis)

        return {
            'detection':      detection_results,
            'temporal_iou':   temporal_iou,
            'boundary':       boundary_metrics,
            'false_positives': fp_analysis,
            'false_negatives': fn_analysis,
            'pred_music_list': pred_music,
        }

    def music_detection_accuracy(self, pred_music):

        results = {}

        for threshold in (0.5, 0.75):
            matched_pred = set()
            matched_gt   = set()

            for i, gt in enumerate(self.gt_music):
                best_iou, best_j = 0.0, -1
                for j, pred in enumerate(pred_music):
                    if j in matched_pred:
                        continue
                    iou = self.calculate_iou(gt, pred)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= threshold:
                    matched_gt.add(i)
                    matched_pred.add(best_j)

            tp = len(matched_gt)
            fp = len(pred_music) - len(matched_pred)
            fn = len(self.gt_music) - tp

            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            results[threshold] = {
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': precision,
                'recall':    recall,
                'f1':        f1,
                'matched_gt':   len(matched_gt),
                'matched_pred': len(matched_pred),
            }

            thr_str = f"IoU >= {threshold}"
            print(f"\n  [{thr_str}]")
            print(f"    TP={tp}  FP={fp}  FN={fn}")
            print(f"    Precision : {precision:7.4f}%  ({tp}/{tp+fp})")
            print(f"    Recall    : {recall:7.4f}%  ({tp}/{tp+fn})")
            print(f"    F1-Score  : {f1:7.4f}%")

        return results 

    def temporal_iou_analysis(self, pred_music):
        ious = []
        for gt in self.gt_music:
            best_iou = max(
                (self.calculate_iou(gt, pred) for pred in pred_music),
                default=0.0
            )
            ious.append(best_iou)

        mean_iou   = np.mean(ious)   if ious else 0
        median_iou = np.median(ious) if ious else 0
        iou_80 = sum(1 for v in ious if v >= 0.8)
        iou_90 = sum(1 for v in ious if v >= 0.9)

        print(f"\n  Mean IoU   : {mean_iou:.4f}")
        print(f"  Median IoU : {median_iou:.4f}")
        n = len(ious)
        print(f"  IoU >= 0.8 : {iou_80}/{n} ({iou_80/n*100:.2f}%)")
        print(f"  IoU >= 0.9 : {iou_90}/{n} ({iou_90/n*100:.2f}%)")

        return {
            'mean': mean_iou, 'median': median_iou,
            'iou_80_pct': iou_80 / n * 100 if n else 0,
            'iou_90_pct': iou_90 / n * 100 if n else 0,
            'per_gt': ious,
        }


    def boundary_evaluation(self, pred_music, tolerance=5):
        gt_ends   = [b['end'] for b in self.gt_music]
        pred_ends = [b['end'] for b in pred_music]

        correct_pred = sum(
            1 for pb in pred_ends if any(abs(pb - gb) <= tolerance for gb in gt_ends)
        )
        correct_gt = sum(
            1 for gb in gt_ends if any(abs(gb - pb) <= tolerance for pb in pred_ends)
        )

        precision = correct_pred / len(pred_ends) * 100 if pred_ends else 0
        recall    = correct_gt  / len(gt_ends)   * 100 if gt_ends  else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  Tolerance  : ±{tolerance}s")
        print(f"  Precision  : {precision:.4f}%  ({correct_pred}/{len(pred_ends)})")
        print(f"  Recall     : {recall:.4f}%  ({correct_gt}/{len(gt_ends)})")
        print(f"  F1-Score   : {f1:.4f}%")

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def false_positive_analysis(self, pred_music, all_predicted):
        false_positives = []
        for pred in pred_music:
            pred_dur = pred['end'] - pred['start']
            max_overlap = max(
                (max(0, min(gt['end'], pred['end']) - max(gt['start'], pred['start']))
                 for gt in self.gt_music),
                default=0
            )
            if pred_dur > 0 and max_overlap / pred_dur < 0.3:
                false_positives.append({
                    'start': pred['start'], 'end': pred['end'], 'duration': pred_dur
                })

        fp_count = len(false_positives)
        fp_rate  = fp_count / len(pred_music) * 100 if pred_music else 0
        print(f"\n  Total FP : {fp_count}/{len(pred_music)} ({fp_rate:.4f}%)")
        for i, fp in enumerate(false_positives[:10], 1):
            print(f"    [{i}] {self.fmt(fp['start'])} ~ {self.fmt(fp['end'])} ({fp['duration']:.1f}s)")

        return {'count': fp_count, 'rate': fp_rate, 'details': false_positives}


    def false_negative_analysis(self, pred_music, all_predicted):
        false_negatives = []
        for gt in self.gt_music:
            gt_dur = gt['end'] - gt['start']
            max_overlap, best_type = 0, None
            for pred in all_predicted:
                ov = max(0, min(gt['end'], pred['end']) - max(gt['start'], pred['start']))
                if ov > max_overlap:
                    max_overlap, best_type = ov, pred['type']
            if gt_dur > 0 and (max_overlap / gt_dur < 0.3 or best_type != 'MUSIC'):
                false_negatives.append({
                    'start': gt['start'], 'end': gt['end'], 'duration': gt_dur,
                    'predicted': best_type or 'Not Detected'
                })

        fn_count = len(false_negatives)
        fn_rate  = fn_count / len(self.gt_music) * 100 if self.gt_music else 0
        print(f"\n  Total FN : {fn_count}/{len(self.gt_music)} ({fn_rate:.4f}%)")
        for i, fn in enumerate(false_negatives[:10], 1):
            print(f"    [{i}] {self.fmt(fn['start'])} ~ {self.fmt(fn['end'])} "
                  f"({fn['duration']:.1f}s)  [Predicted: {fn['predicted']}]")

        return {'count': fn_count, 'rate': fn_rate, 'details': false_negatives}


    def print_summary(self, detection, temporal, boundary, fp, fn):
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION SUMMARY : [{self.broadcaster}] {self.date_str}")
        print(f"{'='*70}")


        print(f"\n{'─'*50}")
        print(f"  {'Metric':<18} {'IoU@0.50':>10}  {'IoU@0.75':>10}")
        print(f"{'─'*50}")
        for metric, key in [('Precision (%)', 'precision'),
                             ('Recall    (%)', 'recall'),
                             ('F1-Score  (%)', 'f1')]:
            v50 = detection[0.50][key]
            v75 = detection[0.75][key]
            print(f"  {metric:<18} {v50:>10.2f}  {v75:>10.2f}")
        print(f"{'─'*50}")
        for label, key in [('TP', 'tp'), ('FP', 'fp'), ('FN', 'fn')]:
            print(f"  {label:<18} {detection[0.50][key]:>10}  {detection[0.75][key]:>10}")
        print(f"{'─'*50}")

        print(f"\n  Temporal IoU  — Mean: {temporal['mean']:.4f} | "
              f"IoU≥0.8: {temporal['iou_80_pct']:.1f}%")
        print(f"  Boundary F1   — {boundary['f1']:.2f}%  "
              f"(P={boundary['precision']:.2f}%  R={boundary['recall']:.2f}%)")
        print(f"  False Positives: {fp['count']}  ({fp['rate']:.2f}%)")
        print(f"  False Negatives: {fn['count']}  ({fn['rate']:.2f}%)")
        print(f"\n{'='*70}\n")


    def save_results(self, results, output_file):
        det = results['detection']
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Music Block Evaluation Results: [{self.broadcaster}] {self.date_str}\n\n")
            f.write("## Detection Performance\n")
            f.write(f"{'Metric':<20} {'IoU@0.50':>10}  {'IoU@0.75':>10}\n")
            f.write(f"{'-'*44}\n")
            for metric, key in [('Precision (%)', 'precision'),
                                 ('Recall    (%)', 'recall'),
                                 ('F1-Score  (%)', 'f1')]:
                f.write(f"{metric:<20} {det[0.50][key]:>10.4f}  {det[0.75][key]:>10.4f}\n")
            f.write(f"{'-'*44}\n")
            for label, key in [('TP', 'tp'), ('FP', 'fp'), ('FN', 'fn')]:
                f.write(f"{label:<20} {det[0.50][key]:>10}  {det[0.75][key]:>10}\n")
            f.write("\n")
            temp  = results['temporal_iou']
            bound = results['boundary']
            fp    = results['false_positives']
            fn_r  = results['false_negatives']

            f.write("## Temporal IoU\n")
            f.write(f"Mean IoU   : {temp['mean']:.4f}\n")
            f.write(f"Median IoU : {temp['median']:.4f}\n\n")

            f.write("## Boundary Metrics (±5s)\n")
            f.write(f"Precision : {bound['precision']:.4f}%\n")
            f.write(f"Recall    : {bound['recall']:.4f}%\n")
            f.write(f"F1-Score  : {bound['f1']:.4f}%\n\n")

            f.write("## Error Analysis\n")
            f.write(f"False Positives : {fp['count']} ({fp['rate']:.4f}%)\n")
            f.write(f"False Negatives : {fn_r['count']} ({fn_r['rate']:.4f}%)\n\n")

            if fp['details']:
                f.write("### False Positive Details\n")
                for i, item in enumerate(fp['details'], 1):
                    f.write(f"  [{i}] {self.fmt(item['start'])} ~ {self.fmt(item['end'])} "
                            f"({item['duration']:.1f}s)\n")
                f.write("\n")

            if fn_r['details']:
                f.write("### False Negative Details\n")
                for i, item in enumerate(fn_r['details'], 1):
                    f.write(f"  [{i}] {self.fmt(item['start'])} ~ {self.fmt(item['end'])} "
                            f"({item['duration']:.1f}s)  [Predicted: {item['predicted']}]\n")

        print(f"💾 Results saved: {output_file}")

    def plot_results(self, results, output_path=None):
        fig = plt.figure(figsize=(22, 12))
        gs  = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.35)
        ax_prf  = fig.add_subplot(gs[0, 0:2]) 
        ax_tiou = fig.add_subplot(gs[0, 2])
        ax_bnd  = fig.add_subplot(gs[0, 3])
        ax_err  = fig.add_subplot(gs[1, 0])
        ax_song = fig.add_subplot(gs[1, 1:])
        ax_tl   = fig.add_subplot(gs[2, :])

        self.plot_prf_grouped(ax_prf,  results) 
        self.plot_temporal_iou(ax_tiou, results)
        self.plot_boundary_metrics(ax_bnd, results)
        self.plot_error_analysis(ax_err, results)
        self.plot_per_song_iou(ax_song, results)
        self.plot_timeline(ax_tl, results)

        plt.suptitle(f'Music Block Evaluation: [{self.broadcaster}] {self.date_str}',
                     fontsize=18, fontweight='bold', y=1.01)

        if output_path is None:
            output_path = f"{self.date_str}_music_evaluation.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {output_path}")
        plt.close()


    def plot_prf_grouped(self, ax, results):
        """IoU@0.5 와 IoU@0.75 두 기준의 Precision / Recall / F1 grouped bar"""
        det   = results['detection']
        x     = np.arange(3) 
        width = 0.32
        labels = ['Precision', 'Recall', 'F1-Score']

        vals_50 = [det[0.50]['precision'], det[0.50]['recall'], det[0.50]['f1']]
        vals_75 = [det[0.75]['precision'], det[0.75]['recall'], det[0.75]['f1']]

        bars1 = ax.bar(x - width/2, vals_50, width,
                       label='IoU ≥ 0.50', color='#3498db', alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + width/2, vals_75, width,
                       label='IoU ≥ 0.75', color='#e67e22', alpha=0.85, edgecolor='white')

        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Detection P / R / F1', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        info = (f"IoU@0.5  — TP:{det[0.50]['tp']}  FP:{det[0.50]['fp']}  FN:{det[0.50]['fn']}\n"
                f"IoU@0.75 — TP:{det[0.75]['tp']}  FP:{det[0.75]['fp']}  FN:{det[0.75]['fn']}")
        ax.text(0.01, 0.01, info, transform=ax.transAxes,
                fontsize=8, va='bottom', color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', fc='#f8f9fa', ec='#cccccc'))

    def plot_temporal_iou(self, ax, results):
        temp = results['temporal_iou']
        vals   = [temp['mean']*100, temp['iou_80_pct'], temp['iou_90_pct']]
        labels = ['Mean IoU', 'IoU≥0.8', 'IoU≥0.9']
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.set_title('Temporal IoU', fontsize=13, fontweight='bold')
        ax.set_ylabel('(%)', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    def plot_boundary_metrics(self, ax, results):
        bound  = results['boundary']
        vals   = [bound['precision'], bound['recall'], bound['f1']]
        labels = ['Precision', 'Recall', 'F1']
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.set_title('Boundary Accuracy (±5s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('(%)', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    def plot_error_analysis(self, ax, results):
        tp = results['detection'][0.50]['tp']
        fp = results['false_positives']['count']
        fn = results['false_negatives']['count']
        sizes  = [tp, fp, fn]
        labels = [f'TP\n({tp})', f'FP\n({fp})', f'FN\n({fn})']
        colors  = ['#2ecc71', '#e74c3c', '#f39c12']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            explode=(0.05, 0.05, 0.05)
        )
        for t in texts:      t.set_fontsize(10)
        for t in autotexts:  t.set_color('white'); t.set_fontsize(9); t.set_fontweight('bold')
        ax.set_title('Error Distribution', fontsize=13, fontweight='bold')


    def plot_per_song_iou(self, ax, results):
        pred_music = results.get('pred_music_list', [])
        song_ious  = [
            max((self.calculate_iou(gt, p) for p in pred_music if p['type'] == 'MUSIC'),
                default=0.0)
            for gt in self.gt_music
        ]
        x      = np.arange(len(song_ious))
        colors = ['#27ae60' if v >= 0.75 else '#f39c12' if v >= 0.5 else '#e74c3c'
                  for v in song_ious]
        bars = ax.bar(x, song_ious, color=colors, alpha=0.85, edgecolor='white')
        for bar, iou in zip(bars, song_ious):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{iou:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.axhline(0.75, color='#27ae60', linestyle='--', alpha=0.5, lw=2, label='≥0.75 Good')
        ax.axhline(0.50, color='#f39c12', linestyle='--', alpha=0.5, lw=2, label='≥0.50 OK')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in x], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('GT Song Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('IoU Score', fontsize=11, fontweight='bold')
        ax.set_title('Per-Song Detection IoU', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    def plot_timeline(self, ax, results):
        total_dur  = max(gt['end'] for gt in self.gt_music) if self.gt_music else 7200
        pred_music = results.get('pred_music_list', [])
        fp_details = results['false_positives']['details']
        fn_details = results['false_negatives']['details']

        y, h = 0.5, 0.6

        for i, gt in enumerate(self.gt_music):
            ax.add_patch(plt.Rectangle(
                (gt['start'], y - h/2), gt['end'] - gt['start'], h,
                lw=2.5, edgecolor='#34495e', facecolor='none',
                linestyle='--', alpha=0.6,
                label='Ground Truth (GT)' if i == 0 else ''
            ))

        tp_cnt = fp_cnt = 0
        for pred in pred_music:
            if pred['type'] != 'MUSIC':
                continue
            is_fp = any(abs(pred['start'] - fp['start']) < 1 and abs(pred['end'] - fp['end']) < 1
                        for fp in fp_details)
            color  = '#e74c3c' if is_fp else '#27ae60'
            label  = ('False Positive (FP)' if is_fp and fp_cnt == 0 else
                      'Detected (TP)'       if not is_fp and tp_cnt == 0 else '')
            if is_fp: fp_cnt += 1
            else:     tp_cnt += 1
            ax.barh(y, pred['end'] - pred['start'], left=pred['start'],
                    height=h, color=color, edgecolor='none', alpha=0.75, label=label)

        ax.set_xlim(0, total_dur * 1.02)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Time (MM:SS)', fontsize=12, fontweight='bold')

        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{int(x//60)}:{int(x%60):02d}")
        )
        ax.set_title(
            f"Broadcast Timeline  |  GT:{len(self.gt_music)} songs  "
            f"TP:{tp_cnt}  FP:{fp_cnt}  FN:{len(fn_details)}",
            fontsize=12, fontweight='bold', pad=10
        )
        ax.grid(axis='x', alpha=0.25, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles, labels_l = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_l, handles))
        if fn_details:
            fn_patch = mpatches.Patch(
                facecolor='none', edgecolor='#34495e', linestyle='--', lw=2.5,
                label='Missed (FN - GT only)'
            )
            by_label['Missed (FN - GT only)'] = fn_patch
        ax.legend(list(by_label.values()), list(by_label.keys()),
                  loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=4, fontsize=10, framealpha=0.95)

    def fmt(self, s):
        return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_music_blocks.py <Broadcast> <YYYYMMDD>")
        print("Example: python evaluate_music_blocks.py KBS 20260101")
        print("Example: python evaluate_music_blocks.py SBS 20241120")
        sys.exit(1)

    broadcaster = sys.argv[1].upper()
    date_str    = sys.argv[2]

    base_dir = os.path.join(OUTPUT_DIR, date_str)

    file_configs = [
        {"filename": f"{date_str}-blocks.csv",            "suffix": "base",  "type": "Base Blocks"},
        {"filename": f"{date_str}-selection_music.csv",   "suffix": "music",   "type": "csv(Selection Music)"},
    ]

    available_files = [
        {**cfg, 'filepath': os.path.join(base_dir, cfg['filename'])}
        for cfg in file_configs
        if os.path.exists(os.path.join(base_dir, cfg['filename']))
    ]

    if not available_files:
        print(f"❌ No CSV files found in {base_dir}")
        sys.exit(1)

    evaluator = MusicEvaluator(broadcaster, date_str)

    print(f"\n{'='*70}")
    print(f"🎵 Music Block Evaluation: [{broadcaster}] {date_str}")
    print(f"{'='*70}")
    print(f"\n📂 Found {len(available_files)} file(s):")
    for i, cfg in enumerate(available_files, 1):
        print(f"  [{i}] {cfg['filename']} ({cfg['type']})")

    for i, cfg in enumerate(available_files, 1):
        print(f"\n{'#'*70}")
        print(f"# [{i}/{len(available_files)}] {cfg['type']}")
        print(f"{'#'*70}")

        predicted_blocks = evaluator.load_predicted_blocks(cfg['filepath'])
        results = evaluator.evaluate_all(predicted_blocks)

        txt_out   = os.path.join(base_dir, f"{date_str}-music-eval-{cfg['suffix']}.txt")
        graph_out = os.path.join(base_dir, f"{date_str}_music_eval_{cfg['suffix']}.png")

        evaluator.save_results(results, txt_out)
        evaluator.plot_results(results, graph_out)

        print(f"\n✅ Done [{i}/{len(available_files)}]")
        print(f"   Text : {txt_out}")
        print(f"   Graph: {graph_out}")

    print(f"\n{'='*70}")
    print(f"🎉 All evaluations completed for [{broadcaster}] {date_str}")
    print(f"{'='*70}\n")