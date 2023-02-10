'''
	python_logger.py
'''


import logging
_logger = logging.getLogger(__name__)


# there are 5 standard levels indicating the severity of events.
Logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

# By default, the logging module logs the messages with 
# a severity level of WARNING or above. 
# You can change that by configuring the logging
# module to log events of all levels if you want


'''
Commonly used parameters for basicConfig() are the following:

level: The root logger will be set to the specified severity level.
filename: This specifies the file.
filemode: If filename is given, the file is opened in this mode. The default is a, which means append.
format: This is the format of the log message.

'''
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')



import logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')


logging_filename = 'processing.log'
logging_format = '%(levelname)s - %(asctime)s - %(message)s'
logging.basicConfig(filename=logging_filename, filemode='a',format=logging_format, datefmt='%d-%b-%y %H:%M:%S')
