# 📌 Django REST Framework (DRF) Notes

## 🔹 Introduction
Django REST Framework (DRF) is a powerful toolkit for building Web APIs in Django. It provides features such as authentication, serialization, and viewsets to simplify API development.

---
## 🔹 Main Topics

### 🔹 1. Installation & Setup
#### 📌 Install DRF
```sh
pip install djangorestframework
```
#### 📌 Add to `INSTALLED_APPS`
```python
INSTALLED_APPS = [
    'rest_framework',
]
```

---
### 🔹 2. Serializers
#### 📌 What are Serializers?
Serializers in DRF allow complex data types like Django models to be converted into JSON.

#### 📌 Example of a Serializer
```python
from rest_framework import serializers
from myapp.models import Book

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
```

---
### 🔹 3. Views
#### 📌 Function-Based Views (FBV)
```python
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def hello_world(request):
    return Response({"message": "Hello, World!"})
```

#### 📌 Class-Based Views (CBV)
```python
from rest_framework.views import APIView
from rest_framework.response import Response

class HelloWorldView(APIView):
    def get(self, request):
        return Response({"message": "Hello, World!"})
```

---
### 🔹 4. ViewSets & Routers
#### 📌 Using ViewSets
```python
from rest_framework import viewsets
from myapp.models import Book
from myapp.serializers import BookSerializer

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
```

#### 📌 Registering with a Router
```python
from rest_framework.routers import DefaultRouter
from myapp.views import BookViewSet

router = DefaultRouter()
router.register(r'books', BookViewSet)
```

---
### 🔹 5. Authentication & Permissions
#### 📌 Setting Up Authentication
DRF supports authentication methods like Token-based and JWT.
```python
INSTALLED_APPS += [
    'rest_framework.authtoken',
]
```

#### 📌 Using Permissions
```python
from rest_framework.permissions import IsAuthenticated

class SecureView(APIView):
    permission_classes = [IsAuthenticated]
```

---
### 🔹 6. Testing the API
#### 📌 Using `curl`
```sh
curl -X GET http://127.0.0.1:8000/api/books/
```
#### 📌 Using Postman or DRF Browsable API
Navigate to the API endpoint in the browser for an interactive UI.

---
### ✅ Summary
- DRF simplifies API development in Django.
- Serializers help convert model instances to JSON.
- Views can be FBV, CBV, or ViewSets.
- Authentication and permissions help secure APIs.

🚀 Happy Coding!

