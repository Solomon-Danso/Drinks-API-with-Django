# 📌 Django REST Framework (DRF) Notes

## 🔹 Introduction
Django REST Framework (DRF) is a powerful toolkit for building Web APIs in Django. It provides features such as authentication, serialization, and viewsets to simplify API development.

---
https://www.django-rest-framework.org/


## 🔹 Main Topics

### 🔹 1. Installation & Setup
#### 📌 Install DRF
```python
pip install djangorestframework
pip install markdown       # Markdown support for the browsable API.
pip install django-filter  # Filtering support
```



#### 📌 Create a Django Project
```python
django-admin startproject myproject

python manage.py runserver

```

---



#### 📌 Add to `INSTALLED_APPS`
```python
INSTALLED_APPS = [
    'rest_framework',
]
```

---

#### 📌 Connect to a Database `settings.py`
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',  # Use the MySQL backend
        'NAME': 'HyDjango',                    # The name of your MySQL database
        'USER': 'root',                        # Your MySQL username (default is 'root')
        'PASSWORD': 'HydotTech',                   # Your MySQL password
        'HOST': 'localhost',                   # Set to your MySQL server's host (localhost in this case)
        'PORT': '3306',                        # Default MySQL port
    }
}


```

---

#### 📌 Run the commands 
```python
python manage.py migrate 
python manage.py createsuperuser --username admin --email admin@example.com


```



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
---

# 🔹 Practical Examples 

### 📌 Create a Models
```python
from django.db import models
from pygments.lexers import get_all_lexers
from pygments.styles import get_all_styles

LEXERS = [item for item in get_all_lexers() if item[1]]
LANGUAGE_CHOICES = sorted([(item[1][0], item[0]) for item in LEXERS])
STYLE_CHOICES = sorted([(item, item) for item in get_all_styles()])

class Snippet(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField()
    linenos = models.BooleanField(default=False)
    language = models.CharField(choices=LANGUAGE_CHOICES, default='python', max_length=100)
    style = models.CharField(choices=STYLE_CHOICES, default='friendly', max_length=100)

    class Meta:
        ordering = ['created']

```

---

### 📌 Push the model into the database
```python
python manage.py makemigrations snippets
python manage.py migrate snippets

```

---

### 📌 Create a Serializer
```python
from rest_framework import serializers
from snippets.models import Snippet, LANGUAGE_CHOICES, STYLE_CHOICES

class SnippetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Snippet
        fields = ['id', 'title', 'code', 'linenos', 'language', 'style']


```

---

### 📌 Create a View
```python
from rest_framework import serializers
from snippets.models import Snippet, LANGUAGE_CHOICES, STYLE_CHOICES

class SnippetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Snippet
        fields = ['id', 'title', 'code', 'linenos', 'language', 'style']


```

---






