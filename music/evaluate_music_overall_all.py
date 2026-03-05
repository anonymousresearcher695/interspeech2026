import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from config import OUTPUT_DIR

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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


def load_csv(date, method, broadcaster=None, base_dir=OUTPUT_DIR):
    if broadcaster:
        date_dir = os.path.join(base_dir, broadcaster.lower(), date)
    else:
        date_dir = os.path.join(base_dir, date)
    csv_file = f"{date}-selection_music.csv" if method == 'srt' else f"{date}-blocks.csv"
    csv_path = os.path.join(date_dir, csv_file)
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    pred_music = []
    if 'block_type' in df.columns:
        for _, row in df.iterrows():
            if row['block_type'].strip() == 'MUSIC':
                pred_music.append({'start': float(row['start']), 'end': float(row['end'])})
    else:
        for _, row in df.iterrows():
            pred_music.append({'start': float(row['start']), 'end': float(row['end'])})
    return pred_music


def calculate_iou(gt, pred):
    overlap = max(0, min(gt['end'], pred['end']) - max(gt['start'], pred['start']))
    if overlap == 0:
        return 0.0
    union = (gt['end'] - gt['start']) + (pred['end'] - pred['start']) - overlap
    return overlap / union if union > 0 else 0.0


def evaluate_single(gt, pred):
    if not pred:
        return 0, 0, len(gt)
    matched_pred = set()
    tp = 0
    for gt_item in gt:
        best_iou, best_idx = 0.0, -1
        for i, pred_item in enumerate(pred):
            if i in matched_pred:
                continue
            iou = calculate_iou(gt_item, pred_item)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= 0.5:
            tp += 1
            if best_idx >= 0:
                matched_pred.add(best_idx)
    return tp, len(pred) - len(matched_pred), len(gt) - tp



def evaluate_method(broadcaster, dates, method):
    all_tp = all_fp = all_gt = all_pred = 0
    broadcaster_gt = GROUND_TRUTH_MUSIC.get(broadcaster, {})
    for date in dates:
        gt   = broadcaster_gt.get(date)
        if not gt:
            continue
        pred = load_csv(date, method, broadcaster=broadcaster) or []
        tp, fp, fn = evaluate_single(gt, pred)
        all_tp   += tp
        all_fp   += fp
        all_gt   += len(gt)
        all_pred += len(pred)
    precision = (all_tp / all_pred * 100) if all_pred > 0 else 0
    recall    = (all_tp / all_gt   * 100) if all_gt   > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {'precision': precision / 100, 'recall': recall / 100, 'f1': f1 / 100}


def evaluate_method_all(method):
    all_tp = all_fp = all_gt = all_pred = 0
    for broadcaster, date_dict in GROUND_TRUTH_MUSIC.items():
        for date, gt in date_dict.items():
            pred = load_csv(date, method, broadcaster=broadcaster)
            if not gt or not pred:
                continue
            tp, fp, fn = evaluate_single(gt, pred)
            all_tp   += tp
            all_fp   += fp
            all_gt   += len(gt)
            all_pred += len(pred)
    precision = (all_tp / all_pred * 100) if all_pred > 0 else 0
    recall    = (all_tp / all_gt   * 100) if all_gt   > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {'precision': precision / 100, 'recall': recall / 100, 'f1': f1 / 100}


def evaluate_per_date(method):
    per_date_f1 = {}
    for broadcaster, date_dict in GROUND_TRUTH_MUSIC.items():
        for date, gt in date_dict.items():
            key = f"{broadcaster}_{date[4:]}"  # ex) SBS_1120
            pred = load_csv(date, method, broadcaster=broadcaster)
            if not gt or not pred:
                per_date_f1[key] = 0.0
                continue
            tp, fp, fn = evaluate_single(gt, pred)
            precision = (tp / len(pred) * 100) if pred else 0
            recall    = (tp / len(gt)   * 100) if gt   else 0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            per_date_f1[key] = f1
    return per_date_f1


def plot_all(results, output_file='music_detection_all.png'):
    """
    Row 0 : Overall Performance Comparison
    Row 1 : Per-broadcaster Precision / Recall / F1
    Row 2 : Per-Date F1  BASE | SRT
    """
    # MBC 데이터 없으면 그래프에서 제외
    broadcasters = [b for b in results if results[b]['base']['precision'] > 0
                    or results[b]['srt']['precision'] > 0]

    baseline_color = '#5A9BD5'
    audio_color    = '#E74C3C'
    base_f1_color  = '#5B9BD5'
    srt_f1_color   = '#2ECC71'

    # ── 데이터 ──
    all_base = evaluate_method_all('base')
    all_srt  = evaluate_method_all('srt')
    metrics  = ['Precision', 'Recall', 'F1-Score']
    ov_base  = [all_base['precision']*100, all_base['recall']*100, all_base['f1']*100]
    ov_srt   = [all_srt['precision']*100,  all_srt['recall']*100,  all_srt['f1']*100]

    bp = [results[b]['base']['precision'] for b in broadcasters]
    br = [results[b]['base']['recall']    for b in broadcasters]
    bf = [results[b]['base']['f1']        for b in broadcasters]
    ap = [results[b]['srt']['precision']  for b in broadcasters]
    ar = [results[b]['srt']['recall']     for b in broadcasters]
    af = [results[b]['srt']['f1']         for b in broadcasters]

    base_f1_map  = evaluate_per_date('base')
    srt_f1_map   = evaluate_per_date('srt')
    all_dates    = list(base_f1_map.keys())
    short_dates  = all_dates  # 이미 SBS_1120 형식
    base_f1_vals = [base_f1_map[d] for d in all_dates]
    srt_f1_vals  = [srt_f1_map[d]  for d in all_dates]

    # ── Figure ──
    fig = plt.figure(figsize=(18, 16))
    gs  = fig.add_gridspec(3, 6, hspace=0.55, wspace=0.4)

    ax_overall = fig.add_subplot(gs[0, 1:5])
    ax_p  = fig.add_subplot(gs[1, 0:2])
    ax_r  = fig.add_subplot(gs[1, 2:4])
    ax_f1 = fig.add_subplot(gs[1, 4:6])
    ax_bd = fig.add_subplot(gs[2, 0:3])
    ax_sd = fig.add_subplot(gs[2, 3:6])

    width = 0.35

    # Row 0: Overall
    x  = np.arange(len(metrics))
    b1 = ax_overall.bar(x - width/2, ov_base, width, label='Baseline',          color=baseline_color)
    b2 = ax_overall.bar(x + width/2, ov_srt,  width, label='Audio Acoustic Features', color=audio_color)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax_overall.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_overall.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax_overall.set_title('Overall Performance Comparison', fontsize=13, fontweight='bold')
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(metrics, fontsize=11)
    ax_overall.set_ylim(0, 115)
    ax_overall.legend(fontsize=10, loc='upper left')
    ax_overall.grid(axis='y', alpha=0.3, linestyle='--')
    ax_overall.spines['top'].set_visible(False)
    ax_overall.spines['right'].set_visible(False)

    # Row 1: Per-broadcaster
    xb = np.arange(len(broadcasters))
    for ax, base_vals, audio_vals, ylabel in zip(
        [ax_p, ax_r, ax_f1],
        [bp, br, bf],
        [ap, ar, af],
        ['Precision', 'Recall', 'F1'],
    ):
        bars1 = ax.bar(xb - width/2, base_vals,  width, label='Baseline',         color=baseline_color)
        bars2 = ax.bar(xb + width/2, audio_vals, width, label='Audio Acoustic Feature', color=audio_color)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xlabel('Broadcaster', fontsize=10, fontweight='bold')
        ax.set_xticks(xb)
        ax.set_xticklabels(broadcasters, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles, labels = ax_p.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.655),
               ncol=2, frameon=False, fontsize=10)

    # Row 2: Per-date F1
    for ax, vals, color, title in [
        (ax_bd, base_f1_vals, base_f1_color, 'Per-Date F1: BASE'),
        (ax_sd, srt_f1_vals,  srt_f1_color,  'Per-Date F1: SRT'),
    ]:
        bars = ax.bar(short_dates, vals, color=color, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score (%)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.set_xticks(range(len(short_dates)))
        ax.set_xticklabels(short_dates, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Music Detection Performance Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Graph saved: {output_file}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    broadcaster_groups = {
        'SBS': list(GROUND_TRUTH_MUSIC['SBS'].keys()), # -> you can add more programs
        'MBC': list(GROUND_TRUTH_MUSIC['MBC'].keys()),
        'KBS': list(GROUND_TRUTH_MUSIC['KBS'].keys()),
    }

    print("🎵 Evaluating Music Detection Performance...")
    print(f"{'='*60}\n")

    results = {}
    for broadcaster, dates in broadcaster_groups.items():
        print(f"Evaluating {broadcaster}  ({len(dates)} dates)...")
        base_result = evaluate_method(broadcaster, dates, 'base')
        srt_result  = evaluate_method(broadcaster, dates, 'srt')
        results[broadcaster] = {'base': base_result, 'srt': srt_result}
        print(f"  Baseline : P={base_result['precision']:.3f}, "
              f"R={base_result['recall']:.3f}, F1={base_result['f1']:.3f}")
        print(f"  Audio FP : P={srt_result['precision']:.3f},  "
              f"R={srt_result['recall']:.3f},  F1={srt_result['f1']:.3f}")
        print()

    plot_all(results, 'music_detection_all.png')
    print("\nDone!  →  music_detection_all.png")