SUMMER_MODEL = {
    "algorithm": "random_forest",
    "params": {
        "criterion": [
            "mse"
        ],
        "max_depth": [
            6
        ],
        "max_features": [
            "sqrt"
        ],
        "n_estimators": [
            30
        ],
        "random_state": [10]
    },
    "optimization_metric": "r2",
    "horizon": "1w",
    "training_data": {
        "endogenous": "real_energy_prod",
        "exogenous": [
            "air_temp",
            "dhi",
            "dni",
            "ghi",
            "cloud_opacity",
            "zenith"
        ],
        "time_range_start": "2020-08-09T00:00:00Z",
        "time_range_end": "2020-08-31T00:00:00Z"
    },
    # "derived_features": {
        # "ghi_std6": {
        #     "transform": "rolling",
        #     "input": "ghi",
        #     "params": {
        #         "window": "6h",
        #         "window_function": "variance"
        #     }
        # },
        # "prod_sh24": {
        #     "transform": "shift",
        #     "input": "real_energy_prod",
        #     "params": {
        #         "periods": 24
        #     }
        # },
        # "air_temp_std6": {
        #     "transform": "rolling",
        #     "input": "air_temp",
        #     "params": {
        #         "window": "6h",
        #         "window_function": "variance"
        #     }
        # },
        # "dni_min12": {
        #     "transform": "rolling",
        #     "input": "dni",
        #     "params": {
        #         "window": "12h",
        #         "window_function": "min"
        #     }
        # },
        # "dni_diff_1": {
        #     "transform": "diff",
        #     "input": "dni",
        #     "params": {
        #         "periods": 1
        #     }
        # },
        # "dni_diff_24": {
        #     "transform": "diff",
        #     "input": "dni",
        #     "params": {
        #         "periods": 24
        #     }
        # },
        # "zenith_std12": {
        #     "transform": "rolling",
        #     "input": "zenith",
        #     "params": {
        #         "window": "12h",
        #         "window_function": "variance"
        #     }
        # },
        # "dhi_max6": {
        #     "transform": "rolling",
        #     "input": "dhi",
        #     "params": {
        #         "window": "6h",
        #         "window_function": "max"
        #     }
        # },
        # "prod_ewm": {
        #     "transform": "ewm",
        #     "input": "real_energy_prod",
        #     "params": {
        #         "window_function": "mean",
        #         "com": 0.5
        #     }
        # },
        # "produkcja_std6": {
        #     "transform": "rolling",
        #     "input": "real_energy_prod",
        #     "params": {
        #         "window": "6h",
        #         "window_function": "variance"
        #     }
        # },
        # "ghi_shift_24": {
        #     "transform": "shift",
        #     "input": "ghi",
        #     "params": {
        #         "periods": 24
        #     }
        # },
        # "ghi_diff_1": {
        #     "transform": "diff",
        #     "input": "ghi",
        #     "params": {
        #         "periods": 1
        #     }
        # },
        # "cloud_opacity_min6": {
        #     "transform": "rolling",
        #     "input": "cloud_opacity",
        #     "params": {
        #         "window": "6h",
        #         "window_function": "min"
        #     }
        # },
        # "dhi_diff_1": {
        #     "transform": "diff",
        #     "input": "dhi",
        #     "params": {
        #         "periods": 1
        #     }
        # },
        # "zenith_max12": {
        #     "transform": "rolling",
        #     "input": "zenith",
        #     "params": {
        #         "window": "12h",
        #         "window_function": "max"
        #     }
        # },
        # "hour_mean": {
        #     "transform": "y_hour_mean",
        #     "input": None
        # },
        # "hour_mean_diff_1": {
        #     "transform": "diff",
        #     "input": "hour_mean",
        #     "params": {
        #         "periods": 1
        #     }
        # },
        # "hour_mean_diff_24": {
        #     "transform": "diff",
        #     "input": "hour_mean",
        #     "params": {
        #         "periods": 24
        #     }
        # },
        # "hour_mean_max_12": {
        #     "transform": "rolling",
        #     "input": "hour_mean",
        #     "params": {
        #         "window": "12h",
        #         "window_function": "max"
        #     }
        # }
    # },
    "target_frequency": "1h",
}
