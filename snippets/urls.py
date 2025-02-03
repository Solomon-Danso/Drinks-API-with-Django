from django.urls import path
from snippets import views

urlpatterns = [
    path('snippets/', views.snippets_list),
    path('snippets/<int:id>/<str:codeID>', views.snippets_specific),

    path('role/', views.RoleMain),
    path('role/<str:userID>', views.RoleDetail),



]