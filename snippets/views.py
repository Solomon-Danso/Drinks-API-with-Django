from django.http import HttpResponse, JsonResponse
from snippets.models import *
from snippets.serializers import *
from snippets.constants import Constants as Const
from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(['GET', 'POST'])
def snippets_list(request):
    if request.method == Const.Get:
        auth = Const.RoleAuthenticator(request.headers.get('UserID'), "Can_View_Snippets")
        if auth is not None:
            return auth  
        
        snippets = Snippet.objects.all() # Database Operation
        serializer = SnippetSerializer(snippets, many=True) # Serializer Bridge
        return Const.Response(Const.success, "Snippets retrieved successfully", serializer.data)
    
    elif request.method == Const.Post:
        auth = Const.RoleAuthenticator(request.headers.get('UserID'), "Can_View_Snippets")
        if auth is not None:
            return auth  
        
        FreshData = Const.Requester(request)

        FreshData['codeID'] = Const.IdGenerator(8)
        serializer = SnippetSerializer(data=FreshData)
        if serializer.is_valid():
            serializer.save()
            return Const.Response(Const.success, "Snippets Saved successfully")
        else:
             error_message = f"Failed to save snippet: {serializer.errors}"
             Const.Logger(Const.error, error_message)
             return Const.Response(Const.failed, "Snippets Failed")


@api_view(['GET', 'PUT', 'DELETE'])
def snippets_specific(request, id, codeID):
    try:
      snippet = Snippet.objects.get(id=id, codeID=codeID)
    except Snippet.DoesNotExist:
        return Const.Response(Const.failed, "Snippets does not exist")
    
    
    if request.method == Const.Get:
        snp = SnippetSerializer(snippet)
        return Const.Response(Const.success, message="Specific Data", data=snp.data)
    
    
    elif request.method == Const.Update:

        FreshData = Const.Requester(request)

        serializer = SnippetSerializer(snippet,data=FreshData)
        if serializer.is_valid():
            serializer.save()
            return Const.Response(Const.success, "Snippets Updated successfully", data=serializer.data)
        else:
             error_message = f"Failed to save snippet: {serializer.errors}"
             Const.Logger(Const.error, error_message)
             return Const.Response(Const.failed, "Snippets Failed")

    elif request.method == Const.Delete:
        snippet.delete()
        return Const.Response(Const.success, "Snippets Deleted")
    


@api_view(['GET', 'POST'])
def RoleMain(request):
    if request.method == Const.Get:
        snippets = Roles.objects.all() # Database Operation
        serializer = RoleSerializer(snippets, many=True) # Serializer Bridge
        return Const.Response(Const.success, "Role retrieved successfully", serializer.data)
    
    elif request.method == Const.Post:
        FreshData = Const.Requester(request)
        serializer = RoleSerializer(data=FreshData)
        if serializer.is_valid():
            serializer.save()
            return Const.Response(Const.success, "Role Saved successfully")
        else:
             error_message = f"Failed to save snippet: {serializer.errors}"
             Const.Logger(Const.error, error_message)
             return Const.Response(Const.failed, "Role Failed")


@api_view(['GET', 'DELETE'])
def RoleDetail(request, userID):
    try:
      snippet = Roles.objects.get(userID=userID)
    except Roles.DoesNotExist:
        return Const.Response(Const.failed, "Role does not exist")
    
    
    if request.method == Const.Get:
        snp = RoleSerializer(snippet)
        return Const.Response(Const.success, message="Specific Data", data=snp.data)
    
    
   
    elif request.method == Const.Delete:
        Roles.delete()
        return Const.Response(Const.success, "Roles Deleted")




