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
        FreshData['codeID'] = Const.IdGenerator(4)
        serializer = SnippetSerializer(data=FreshData)
        if serializer.is_valid():
            serializer.save()
            return Const.Response(Const.success, "Snippets Saved successfully")
        else:
             error_message = f"Failed to save snippet: {serializer.errors}"
             Const.Logger(Const.error, error_message)
             return Const.Response(Const.failed, "Snippets Failed")
        

@csrf_exempt
def snippets_specific(request, id, codeID):
    try:
      snippet = Snippet.objects.get(id=id, codeID=codeID)
    except Snippet.DoesNotExist:
        return HttpResponse(status=404)
    
    if request.method == Const.Get:
        snp = SnippetSerializer(snippet)
        return Const.Response(Const.success, message="Specific Data", data=snp.data)
    
    
    elif request.method == Const.Update:
        FreshData = JSONParser().parse(request)
        serializer = SnippetSerializer(snippet,data=FreshData)
        if serializer.is_valid():
            serializer.save()
            return Const.Response(Const.success, "Snippets Updated successfully", data=serializer.data)
        else:
             return Const.Response(Const.failed, "Snippets Failed")

    elif request.method == Const.Delete:
        snippet.delete()
        return Const.Response(Const.success, "Snippets Deleted")
    




