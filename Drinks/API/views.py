from .models import Drinks
from .serializers import DrinkSerializer
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

#get all the model
#serialize them
#return json


@api_view(['GET','POST'])
def drink_list(request,format=None):
    if request.method == "GET":
        drinks = Drinks.objects.all()
        serializer = DrinkSerializer(drinks,many=True)
        return Response(serializer.data)

    if request.method == "POST":
        serializer = DrinkSerializer(data = request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,status=status.HTTP_201_CREATED)

#put is the same us update

@api_view(['GET','PUT','DELETE'])
def drink_detail(request,id,format=None):
    #accessing the database by filtering it with an id and wrapping it in try and except which is not compulsory 
    try:
      drink = Drinks.objects.get(pk=id)
    except Drinks.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)


    if request.method == "GET":
        serializer = DrinkSerializer(drink)
        return Response(serializer.data)

    elif request.method == "PUT":
        serializer = DrinkSerializer(drink, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        #If serializer is not valid, return this errors
        return Response(serializer.errors,status=status.HTTP_400_BAD_REQUEST)


    elif request.method == "DELETE":
        drink.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
