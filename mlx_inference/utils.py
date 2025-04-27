import uuid

def get_session_id(headers):
    return headers.get('X-Session-ID', str(uuid.uuid4()))