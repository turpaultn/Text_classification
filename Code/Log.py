import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
# create logger with 'spam_application'
logger = logging.getLogger('Saxo_application')
logger.setLevel(logging.DEBUG) # remove this line for behaviour without info

# create file handler which logs even debug messages
# replace logger by fh in other files to log into the file
# fh = logging.FileHandler('spam.log')
# fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# add the handlers to the logger
# logger.addHandler(fh)
# logger.addHandler(ch)