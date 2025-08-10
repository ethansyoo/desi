# db_connection_test.py

import sys
import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

def test_mongo_connection(connection_string):
    """
    Tests a direct connection to a MongoDB Atlas cluster.
    """
    print("--- Starting MongoDB Connection Test ---")

    if not connection_string:
        print("\nERROR: Please provide the MongoDB connection string as an argument.")
        print("Usage: python db_connection_test.py \"mongodb://user:pass@your_cluster...\"")
        return

    print(f"\nAttempting to connect with string: {connection_string[:30]}...")

    try:
        # Create a new MongoClient instance
        # Using tls=True and tlsCAFile is the most robust method
        client = MongoClient(
            connection_string,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=20000  # 20-second timeout
        )

        # The ping command is a lightweight way to force a connection
        # and verify that the server is responding.
        print("Pinging the database...")
        client.admin.command('ping')

        print("\n✅ SUCCESS: Connection to MongoDB was successful!")
        print("This confirms that your credentials, IP whitelist, and network settings are correct.")

    except ConfigurationError as e:
        print(f"\n❌ CONFIGURATION ERROR: {e}")
        print("Please check that your connection string is formatted correctly.")
        print("It should start with 'mongodb://', not 'mongodb+srv://'.")

    except ConnectionFailure as e:
        print("\n❌ CONNECTION FAILURE: Could not connect to the database.")
        print("This is likely due to one of the following:")
        print("  1. Incorrect username or password in the connection string.")
        print("  2. Your current IP address is not whitelisted in MongoDB Atlas.")
        print("  3. A firewall or network issue is blocking the connection.")
        print("\nFull error details:")
        print(e)

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: An unknown error occurred.")
        print(e)

    finally:
        print("\n--- Test Complete ---")


if __name__ == "__main__":
    # Get the connection string from the command-line arguments
    conn_string = sys.argv[1] if len(sys.argv) > 1 else ""
    test_mongo_connection(conn_string)