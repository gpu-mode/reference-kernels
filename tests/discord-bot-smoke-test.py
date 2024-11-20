from dotenv import load_dotenv
import discord
import logging
import os
import argparse
import asyncio
import re
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

parser = argparse.ArgumentParser(
    description='Smoke Test for the Discord Cluster Bot',
    epilog=f"""
    This script can be used after deployment, or during development, to quickly
    verify that basic functionality of the cluster bot is working correctly.
    It should be run before further testing or usage of the bot.

    Example usage:
      python {os.path.basename(__file__)} https://discord.com/channels/123/456/789
    The URL is the message link for some message that triggered the cluster bot.
    To find this URL: click the 3 dots (...) to the right of the message,
    then click 'Copy Message Link'.
    
    Limitations:
    - The smoke test does not yet work for Modal runs.""",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('message_url', type=str, help='Discord message URL to test')
args = parser.parse_args()

message_id = int(args.message_url.split('/')[-1])

# Client setup with minimal intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Event that signals when async client tests are done
client_tests_done = asyncio.Event()

# Flag set to true if the thread tests pass
thread_tests_passed = False

async def verify_thread_messages():
    """
    Test messages from a Discord thread identified by a message ID.

    Side effect:
        Sets thread_tests_passed to True if all happy path messages are found
    """

    global thread_tests_passed

    required_strings = [
        "Processing `.*` with",
        "GitHub Action triggered! Run ID:",
        "Training completed with status: success",
        ".*```\nLogs.*:",
        "View the full run at:",
    ]

    message_contents = []
    thread_found = False

    # Iterate through guilds to find the thread by message ID
    for guild in client.guilds:
        try:
            # Search for the thread using the message ID
            for channel in guild.text_channels:
                try:
                    message = await channel.fetch_message(message_id)
                    if message.thread:
                        thread_found = True
                        thread = message.thread
                        logger.info(f"Found thread: {thread.name}.")
                        
                        # Fetch messages from the thread
                        message_contents = [
                            msg.content async for msg in thread.history(limit=None)
                        ]
                        break

                except discord.NotFound:
                    continue
                except discord.Forbidden:
                    logger.warning(f"Bot does not have permission to access {channel.name}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching thread: {e}", exc_info=True)

        if thread_found:
            # Already found the thread, so no need to continue to iterate
            # through guilds
            break

    if message_contents:
        all_strings_found = all(
            any(re.match(req_str, contents, re.DOTALL) != None for contents in message_contents)
            for req_str in required_strings
        )

        if all_strings_found:
            thread_tests_passed = True
    else:
        logger.warning("Thread not found!")

    if thread_tests_passed:
        logger.info('All required strings were found in the thread.')
    else:
        logger.warning('Some required string was not found in the thread!')
        logger.info('Thread contents were: ')
        logger.info('\n'.join(f'\t{contents}' for contents in message_contents))

@client.event
async def on_ready():
    await verify_thread_messages()

    # We could add additional tests that use the client here if needed.

    client_tests_done.set()
    await client.close()

if __name__ == '__main__':
    logger.info("Running smoke tests...")

    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error('DISCORD_TOKEN environment variable not set.')
        exit(1)

    client.run(token)

    async def await_client_tests():
        await client_tests_done.wait()

    asyncio.run(await_client_tests())

    if not thread_tests_passed:
        # If other tests are needed, add them above
        #    if (not thread_test_passed) or (not other_test_passed):
        logger.warning("One or more tests failed!")
        sys.exit(1)
    else:
        logger.info('All tests passed!')
        sys.exit(0)