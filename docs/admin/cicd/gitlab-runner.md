(admin-cicd-gitlab-runner)=
# GitLab Runner

1. Click on your project and select Settings.

2. Navigate to Settings and click on CI/CD inside this click on Expand of Runners section

3. Click New project runner.

4. Assign tags to control which tagged jobs will run on this runner.

    tags: pages

5. Enter pdx-tme-002.nvidia.com in the Runner description.

6. Leave everything else blank, click Create runner.

7. Copy the command provided on the GitLab page.

8. SSH to nvidia@pdx-tme-002.nvidia.com.

9. Slack Andrew Schilling for SSH password.

10. Run that command from the step above with sudo.

11. Use the default GitLab instance URL and name for runner (Hit enter twice).

12. Select Docker for executor.

13. Select docker:23.0.6 for the default Docker image.

14. Confirm the new runner is assigned under Assigned project runners.

15. Deselect Instance runners.

16. Run a test build with the updated runner.