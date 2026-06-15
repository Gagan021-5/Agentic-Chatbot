import httpx
import asyncio

async def test_video():
    url = "https://gen.pollinations.ai/video/a-beautiful-sunrise?width=1024&height=576&seed=123"
    print("Fetching:", url)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            print("Status Code:", resp.status_code)
            print("Content-Type:", resp.headers.get("content-type"))
            print("Response Headers:", dict(resp.headers))
            print("Response Content (first 200 chars):", resp.text[:200])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_video())
