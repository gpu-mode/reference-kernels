## Description

Please provide a brief summary of the changes in this pull request.

## Checklist

Before submitting this PR, ensure the following steps have been completed:

- [ ] Run the smoke test on your own server.
  - Run the cluster bot on your server:
    ```bash
    python discord-bot.py
    ```
  - Start a training run by with the slash command `/run`.
    You may need to exercise some judgement about the script and GPU type.
  - Wait for the training run to complete.
  - Copy the URL for the thread started by the cluster bot in response to
    your `/run` message ("Cluster Bot started a thread: ..."):
      - Click on the 3 dots (`...`) to the cluster bot's message.
      - Select *Copy Message Link*.
  - Using the copied URL, run the smoke test:
    ```bash
    python discord-bot-smoke-test.py copied_url
    ```
  - Verify that the smoke test script responds with:
    ```
    All tests passed!
    ```
  For more information on running a cluster bot on your own server, see
  README.md.
