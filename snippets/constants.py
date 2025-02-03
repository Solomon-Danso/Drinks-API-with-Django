from django.http import JsonResponse

class Constants:
    Get = "GET"
    Post = "POST"
    Update = "Update"
    Delete = "Delete"
    success = 200
    failed = 400
    notFound = 404
    serverError = 500
    unauthorized = 401
    paymentRequired = 402


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
    