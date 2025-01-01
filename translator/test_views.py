import pytest
from django.http import JsonResponse
from django.urls import reverse


# Test if index view renders correctly
@pytest.mark.django_db
def test_index_view(client):
    url = reverse("index")
    response = client.get(url)
    assert response.status_code == 200
    assert b"<title>Index</title>" in response.content


# Test if the letter is returned correctly by get_detected_letter
@pytest.mark.django_db
def test_get_detected_letter(client):
    url = reverse("get_detected_letter")
    response = client.get(url)
    assert response.status_code == 200
    assert isinstance(response, JsonResponse)
    assert "letter" in response.json()


# Test if message is returned correctly by get_message
@pytest.mark.django_db
def test_get_message(client):
    url = reverse("get_message")
    response = client.get(url)
    assert response.status_code == 200
    assert isinstance(response, JsonResponse)
    assert "message" in response.json()


# Test if reset_message works
@pytest.mark.django_db
def test_reset_message(client):
    url = reverse("reset_message")
    response = client.post(url)
    assert response.status_code == 200
    assert response.json() == {"status": "Message reset"}


# Test if start_message changes the status to 'started'
@pytest.mark.django_db
def test_start_message(client):
    url = reverse("start_message")
    response = client.post(url)
    assert response.status_code == 200
    assert response.json() == {"status": "Message concatenation started"}


# Test if stop_message changes the status to 'stopped'
@pytest.mark.django_db
def test_stop_message(client):
    url = reverse("stop_message")
    response = client.post(url)
    assert response.status_code == 200
    assert response.json() == {"status": "Message concatenation stopped"}


# Test if video feed returns a valid response (streaming)
@pytest.mark.django_db
def test_video_feed(client):
    url = reverse("video_feed")
    response = client.get(url)
    assert response.status_code == 200
    assert response["Content-Type"] == (
        "multipart/x-mixed-replace; boundary=frame"
    )
