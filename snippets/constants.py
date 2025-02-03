from django.http import JsonResponse
import random
import string
import logging
import os
from datetime import datetime
from rest_framework.parsers import JSONParser


class Constants:
    Get = "GET"
    Post = "POST"
    Update = "PUT"
    Delete = "DELETE"
    success = 200
    failed = 400
    notFound = 404
    serverError = 500
    unauthorized = 401
    paymentRequired = 402
    error = "ERROR"
    info = "INFO"
    debug = "DEBUG"
    warn = "WARNING"


    def Response(Code, message, data=None):

        if(Code == 200):
            status = "success"
        elif(Code == 400):
            status = "failed"
        elif(Code == 404):
            status = "notFound"
        elif(Code == 500):
            status = "serverError"
        elif(Code == 401):
            status = "unauthorized"
        elif(Code == 402):
            status = "paymentRequired"


        response = {
            "status": status,  # Example: "success" or "error"
            "message": message,  # Example: "Request successful" or "Invalid input"
            "data": data  # The actual response data (can be None)
        }
        return JsonResponse(response, status=Code)
    
    def IdGenerator(length):
        return ''.join(random.choices(string.digits, k=length))  # Generates a string of 8 random digits


    def Logger(log_level, message):
        # Create a 'logs' directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Get current date to create a unique log file for each day
        log_filename = datetime.now().strftime('%Y-%m-%d') + '.txt'
        log_filepath = os.path.join('logs', log_filename)

        # Set up the logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Log all levels of messages

        # Create a file handler to write log messages to the file
        file_handler = logging.FileHandler(log_filepath, mode='a')  # Append mode
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter that specifies the format of log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Log the message based on the log level
        if log_level == "ERROR":
            logger.error(message)
        elif log_level == "INFO":
            logger.info(message)
        elif log_level == "DEBUG":
            logger.debug(message)
        elif log_level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)  # Default to INFO level if an invalid log level is passed


    def Requester(request):
        try:
            if request.content_type == 'application/json':
                FreshData = JSONParser().parse(request)
            else:
                FreshData = request.POST.dict()  # Convert form data to dictionary
        except Exception as e:
            Constants.Logger(Constants.error, e)
        
        return FreshData





