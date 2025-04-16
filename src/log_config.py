from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s | %(name)s.%(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },

    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',  # Log rotation
            'level': 'INFO',
            'formatter': 'standard',
            'filename': f'logs/scraper_{current_time}.log'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',  # Log rotation for error logs
            'level': 'ERROR',
            'formatter': 'standard',
            'filename': f'logs/scraper_error_{current_time}.log'
        },
    },

    'loggers': {
        'arize': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False,
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console'],
    }
}
