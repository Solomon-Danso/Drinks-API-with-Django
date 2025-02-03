from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer
from snippets.constants import Constants as Const

@csrf_exempt
def snippets_list(request):
    if request.method == Const.Get:
        snippets = Snippet.objects.all() # Database Operation
        serializer = SnippetSerializer(snippets, many=True) # Serializer Bridge
        return Const.Response(Const.success, "Snippets retrieved successfully", serializer.data)
    
    elif request.method == Const.Post:
        FreshData = JSONParser().parse(request)
        serializer = SnippetSerializer(data=FreshData)
        if serializer.is_valid():
            serializer.save()
            return Const.Response(Const.success, "Snippets Saved successfully")
        else:
             return Const.Response(Const.failed, "Snippets Failed")
        


    




