# ruff: noqa: E501
import os
from datetime import datetime

import psycopg2
import requests
from jinja2 import Template

TOKEN = os.environ.get("DISCORD_DUMMY_TOKEN")


def get_name_from_id(user_id: str) -> str:
    """
    Get Discord global name from USER_ID
    """
    url = f"https://discord.com/api/v10/users/{user_id}"
    headers = {"Authorization": f"Bot {TOKEN}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        user_data = response.json()
        return user_data.get("global_name", user_id)
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
                    <div class="submission{% if submission.is_fastest %} fastest{% endif %}"
                         data-user="{{ submission.user }}"
                         data-time="{{ submission.time }}">
                        {{ submission.user }} - {{ submission.time }}
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
                # Get active leaderboards with their GPU types and submission counts
                cur.execute("""
                    WITH ranked_submissions AS (
                        SELECT
                            l.id,
                            l.name,
                            l.deadline,
                            gt.gpu_type,
                            s.user_id,
                            s.score as time,
                            RANK() OVER (PARTITION BY l.id, gt.gpu_type ORDER BY s.score ASC) as rank
                        FROM leaderboard.leaderboard l
                        JOIN leaderboard.gpu_type gt ON gt.leaderboard_id = l.id
                        LEFT JOIN leaderboard.submission s ON s.leaderboard_id = l.id AND s.gpu_type = gt.gpu_type
                        WHERE l.deadline > NOW()
                    )
                    SELECT
                        id,
                        name,
                        deadline,
                        gpu_type,
                        array_agg(
                            CASE WHEN user_id IS NOT NULL THEN
                                json_build_object(
                                    'user_id', user_id,
                                    'time', time,
                                    'rank', rank
                                )
                            ELSE NULL END
                        ) as submissions
                    FROM ranked_submissions
                    GROUP BY id, name, deadline, gpu_type
                    HAVING COUNT(user_id) > 0
                    ORDER BY deadline ASC;
                """)

                leaderboards = cur.fetchall()
                print(f"Found {len(leaderboards)} active leaderboards with submissions")

                gpu_type_data = {}

                for _lb_id, name, deadline, gpu_type, submissions_json in leaderboards:
                    if gpu_type not in gpu_type_data:
                        gpu_type_data[gpu_type] = {}

                    if submissions_json:
                        gpu_submissions = []
                        for sub in submissions_json:
                            if sub is not None:  # Skip NULL entries from array_agg
                                global_name = get_name_from_id(sub["user_id"])
                                gpu_submissions.append(
                                    {
                                        "user": f"{global_name}",
                                        "time": f"{sub['time']:.9f}",
                                        "is_fastest": sub["rank"] == 1,
                                    }
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
