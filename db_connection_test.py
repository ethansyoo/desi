# db_connection_test.py

import sys
import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

# Check for dnspython
try:
    import dns.resolver
    print("✅ `dnspython` is installed.")
except ImportError:
    print("⚠️ WARNING: `dnspython` is not installed. This is required for 'mongodb+srv://' connection strings.")
    print("Please run: pip install dnspython")

def test_mongo_connection(connection_string):
    """
    Tests a direct connection to a MongoDB Atlas cluster.
    """
    print("\n--- Starting MongoDB Connection Test ---")

    if not connection_string:
        print("\n❌ ERROR: Please provide the MongoDB connection string as an argument.")
        print("Usage: python db_connection_test.py \"mongodb+srv://user:pass@your_cluster...\"")
        return

    print(f"\nAttempting to connect with string: {connection_string[:35]}...")

    try:
        client = MongoClient(
            connection_string,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=20000  # 20-second timeout
        )
        print("Pinging the database...")
        client.admin.command('ping')
        print("\n✅ SUCCESS: Connection to MongoDB was successful!")
        print("This confirms that your credentials, IP whitelist, and network settings are correct.")

    except ConfigurationError as e:
        print(f"\n❌ CONFIGURATION ERROR: {e}")
        print("\n**Troubleshooting Steps:**")
        print("1. **Check String Format**: Ensure your connection string starts with `mongodb+srv://`.")
        print("2. **Install dnspython**: If you haven't already, run `pip install dnspython`.")
        print("3. **Verify Hostname**: Double-check the cluster hostname for typos.")

    except ConnectionFailure as e:
        print("\n❌ CONNECTION FAILURE: Could not connect to the database.")
        print("\n**Troubleshooting Steps:**")
        print("1. **Check IP Access List**: Go to your MongoDB Atlas dashboard -> Network Access and ensure your current IP is added. If you're on a dynamic IP, try adding `0.0.0.0/0` (allow access from anywhere) for testing purposes.")
        print("2. **Verify Credentials**: Double-check the username and password in your connection string.")
        print("3. **Check for Firewalls**: Ensure no firewalls are blocking outbound traffic on port 27017.")
        print("\nFull error details:")
        print(e)

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: An unknown error occurred.")
        print(e)

    finally:
        print("\n--- Test Complete ---")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        conn_string = sys.argv[1]
        test_mongo_connection(conn_string)
    else:
        print("Please provide the connection string as a command-line argument.")