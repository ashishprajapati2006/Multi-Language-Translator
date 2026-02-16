from app import app

# For Vercel serverless
def handler(request):
    return app(request)
