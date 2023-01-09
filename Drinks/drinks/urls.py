
from django.contrib import admin
from django.urls import path
from API import views
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path("drinks/",views.drink_list),
    path("drinks/<int:id>",views.drink_detail)
]
#It will help us to add .json extension
urlpatterns = format_suffix_patterns(urlpatterns)