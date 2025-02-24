# ruff: noqa: E501
import os
from datetime import datetime

import psycopg2
import requests
from jinja2 import Template

TOKEN = os.environ.get("DISCORD_DUMMY_TOKEN")

cached_names = {}


def get_name_from_id(user_id: str) -> str:
    """
    Get Discord global name from USER_ID
    """
    if user_id in cached_names:
        return cached_names[user_id]

    url = f"https://discord.com/api/v10/users/{user_id}"
    headers = {"Authorization": f"Bot {TOKEN}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        user_data = response.json()
        name = user_data.get("global_name", user_id)
        cached_names[user_id] = name
        return name
    else:
        return f"User_{user_id}"


print("Starting leaderboard update script...")

# HTML template
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Active Leaderboards</title>
</head>
<body>
    <div class="timestamp">Last updated: {{ timestamp }}</div>
    <div class="gpu-types-container">
        {% for gpu_type in gpu_types %}
        <div class="gpu-type" data-gpu-type="{{ gpu_type.name }}">
            <h2 class="gpu-type-name">{{ gpu_type.name }}</h2>
            {% for problem in gpu_type.problems %}
            <div class="problem" data-problem-name="{{ problem.name }}">
                <h3 class="problem-name">{{ problem.name }}</h3>
                <div class="problem-deadline">Deadline: {{ problem.deadline }}</div>
                <div class="submissions-list">
                    {% for submission in problem.submissions %}
                    <div class="submission{% if submission.rank == 1 %} first{% elif submission.rank == 2 %} second{% elif submission.rank == 3 %} third{% endif %}"
                         data-user="{{ submission.user }}"
                         data-time="{{ submission.time }}s"
                         {% if submission.rank %}data-rank="{{ submission.rank }}"{% endif %}>
                        {% if submission.rank == 1 %}🥇 {% elif submission.rank == 2 %}🥈 {% elif submission.rank == 3 %}🥉 {% else %}{{ submission.rank }}. {% endif %}{{ submission.user }} - {{ submission.time }}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

# Database connection - use environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")
print("Connecting to database...")


def fetch_leaderboard_data():
    print("Fetching data from database...")
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, deadline
                    FROM leaderboard.leaderboard
                    """
                )

                leaderboards = cur.fetchall()

                # Get active leaderboards with their GPU types and submission counts
                query = """
                    WITH ranked_submissions AS (
                        SELECT
                            s.file_name,
                            s.user_id,
                            s.submission_time,
                            r.score,
                            r.runner,
                            RANK() OVER (ORDER BY r.score ASC) as rank
                        FROM leaderboard.runs r
                        JOIN leaderboard.submission s ON r.submission_id = s.id
                        JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                        WHERE l.name = %s AND r.runner = %s AND NOT r.secret
                            AND r.score IS NOT NULL AND r.passed
                    )
                    SELECT * FROM ranked_submissions
                    ORDER BY %s ASC;
                """

                gpu_type_data = {}
                for (
                    _lb_id,
                    name,
                    deadline,
                ) in leaderboards:
                    cur.execute(
                        "SELECT * from leaderboard.gpu_type where leaderboard_id = %s", [_lb_id]
                    )
                    gpu_types = [x[1] for x in cur.fetchall()]

                    for gpu_type in gpu_types:
                        args = (name, gpu_type, deadline)
                        cur.execute(query, args)
                        submissions = cur.fetchall()

                        print(
                            f"Found {len(submissions)} active submissions in {name} for {gpu_type}"
                        )

                        if len(submissions) > 0:
                            if gpu_type not in gpu_type_data:
                                gpu_type_data[gpu_type] = {}

                            gpu_submissions = []
                            for lb in submissions:
                                user_id = lb[1]
                                time = lb[3]
                                rank = lb[5]
                                global_name = get_name_from_id(user_id)
                                gpu_submissions.append(
                                    {"user": f"{global_name}", "time": f"{time:.9f}", "rank": rank}
                                )

                            # Sort submissions by time
                            gpu_submissions.sort(key=lambda x: float(x["time"]))

                            gpu_type_data[gpu_type][name] = {
                                "name": name,
                                "deadline": deadline.strftime("%Y-%m-%d %H:%M"),
                                "submissions": gpu_submissions,
                            }

                # Convert to final format
                formatted_data = {
                    "gpu_types": [
                        {"name": gpu_type, "problems": list(problems.values())}
                        for gpu_type, problems in gpu_type_data.items()
                    ],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                }

                print("Data fetched successfully")
                return formatted_data
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise


def write_html_file(data):
    try:
        # Define the paths
        paths = [
            "static/leaderboard",
            "docs/static/leaderboard",
            "docs/build/leaderboard",  # Add build directory as well
        ]

        # Create directories if they don't exist
        for path in paths:
            os.makedirs(path, exist_ok=True)
            print(f"Ensured directory exists: {path}")

        # Render the template
        template = Template(TEMPLATE)
        html_content = template.render(**data)

        # Write to all locations
        filename = "table.html"
        for path in paths:
            full_path = os.path.join(path, filename)
            with open(full_path, "w") as f:
                f.write(html_content)
            print(f"Written to {full_path}")

        return True
    except Exception as e:
        print(f"Error writing HTML file: {str(e)}")
        return False


def main():
    try:
        # Generate and save leaderboard
        data = fetch_leaderboard_data()
        if write_html_file(data):
            # Print contents of directories
            for static_dir in [
                "static/leaderboard",
                "docs/static/leaderboard",
                "docs/build/leaderboard",
            ]:
                print(f"\nContents of {static_dir}:")
                for file in os.listdir(static_dir):
                    print(f"- {file}")

    except Exception as e:
        print(f"Error updating leaderboard: {str(e)}")
        raise


if __name__ == "__main__":
    main()
