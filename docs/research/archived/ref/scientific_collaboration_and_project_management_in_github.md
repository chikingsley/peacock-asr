# Scientific Collaboration and Project Management in GitHub

*Cleaned transcript from a raw Medium scrape.*

**Author:** Ryan Abernathey  
**Published:** October 26, 2020  
**Read time:** 8 min

---

This short blog post is a technical follow-up to an earlier post on advising and collaborating during the pandemic.

In that post, Abernathey described his frustration with fragmented collaboration and project-management tools, especially once remote work became the norm during COVID-19. Keeping track of many projects and collaborations requires structure, particularly when team members are not meeting face to face. He had previously used a mix of tools such as Nirvana for personal task tracking and Basecamp for group projects, which left tasks and to-dos spread across multiple systems. His resolution was to track every aspect of his work in GitHub. Between Issues and Projects, he found GitHub to be a flexible and powerful project-management system.

For a few months, his group had been using GitHub not only for version control, but also for managing general scientific research projects. The main reasons for choosing GitHub were:

- **Project management features.** GitHub Issues can track to-do items, while Project Boards can organize issues from different repositories into larger workstreams.
- **Global namespace.** Most collaborators already have GitHub accounts, so people can be tagged or assigned without forcing them to adopt another platform.
- **Cross-references.** GitHub makes it easy to reference issues across repositories and organizations, which matters when research projects depend on software maintained elsewhere.
- **Rich communication.** GitHub Issues support long-form technical discussion, linked documents, images, and equations via Markdown extensions.

Abernathey then outlines how his group organizes scientific work on GitHub.

## Project == Repo

Every scientific project gets its own GitHub repository.

In this workflow, a "project" is a concrete piece of work that roughly corresponds to a single scientific publication. The project is considered complete when the paper is published. A project typically has one lead, usually the lead author, and one or more collaborators including the PI.

Most of their repositories live in the group's GitHub organization, but they do not have to. A repository can live in a personal account or another organization's namespace. That decentralization is part of GitHub's appeal.

By the end of the project, the repository should contain all code needed to reproduce the work. For people looking for a repository template, Abernathey recommends Julius Busecke's "Cookie Cutter Science Project." At the same time, he notes that a project does not need to start with a full codebase. Early on, the repository is often more about tracking work than storing software, and the issue tracker is the main tool for that.

## TODO == Issue

GitHub Issues are the core of the system.

Issues represent small, specific tasks that move the project forward. The goal is to enumerate as much of the project as possible in the issue tracker. That usually does not happen all at once. Instead, the team starts with a few first steps and adds issues as the project evolves.

Open issues represent future work. Closed issues represent completed work. The closed-issue list becomes a running summary of everything already done on the project.

Anyone involved in the project can add issues. If someone has an idea in the middle of the night, they should create an issue. Issues should be assigned to the person responsible for doing the work. Often that is the project lead, but collaborative projects may have many contributors working in parallel.

Issues can be extremely short, such as "Download this data," or very detailed, with long technical discussions in the comments. Abernathey emphasizes that this kind of deep asynchronous discussion is one of GitHub's strengths because it allows technical decisions to happen without requiring a meeting.

## Meeting == Milestone

GitHub Milestones provide the structure for meetings.

Milestones can be used in many ways, but this group found it useful to create a milestone for each team meeting. That gives contributors a concrete target date and helps create momentum, especially in a work-from-home environment.

This also creates a built-in meeting agenda. Each meeting should:

- Review progress on the issues targeted for that milestone, including any challenges or setbacks.
- Brainstorm new issues for future work.
- Decide which issues should be assigned to the next meeting's milestone.

## Everything in One Place

This workflow works easily when someone is involved in only one repository. The challenge appears when supervisors or collaborators are spread across many repositories and organizations.

The key tool for handling that is:

- `https://github.com/issues`

That page shows all issues a person has created, been assigned to, or been mentioned in.

Another useful tool is GitHub Project Boards, which provide a Kanban-style view of task flow:

- `https://docs.github.com/en/free-pro-team@latest/github/managing-your-work-on-github/about-project-boards`

Project Boards can combine issues and milestones from multiple repositories within the same organization to give a high-level view of a large effort. Abernathey notes that his group had not yet incorporated Project Boards into its workflow, but he could see them being useful for tracking milestones and deliverables in large, multi-year, multi-institution grants.

## Integration with Slack and Email

The goal is to keep all work tracked in GitHub, but tasks often originate elsewhere, especially in Slack and email.

Slack had become an important communication tool for the group during the pandemic, but it is a chat app rather than a project-management tool, so important items can easily get lost. To connect Slack and GitHub, they used the GitHub Slack integration:

- `https://slack.github.com`

Their pattern was to create a Slack channel for each project and subscribe it to GitHub notifications using:

```text
/github subscribe <organization>/<repo> comments
```

That way, Slack posts every time there is activity on the repository. They could also open new issues directly from Slack.

For email, they experimented with Fire, a tool that converts forwarded emails into GitHub Issues:

- `https://fire.fundersclub.com`

It was not perfect, but still useful.

## Problems and Challenges

Abernathey concludes that the approach was working well overall. It helped the group stay productive and communicate asynchronously across multiple projects while everyone was remote.

At the same time, he stresses that the system requires commitment. People have to check repositories regularly, participate in detailed issue discussions, and use the issue tracker to shape meeting agendas. The workflow becomes much less effective if people maintain a separate "shadow" system for their real priorities.

He identifies several challenges that remained:

- **Tracking many issues is hard.** Participating in hundreds of discussions makes it easy to lose track of important conversations.
- **Some colleagues will not participate.** Senior collaborators in particular may resist adopting a new tool, forcing the team to split coordination across GitHub and email.
- **Writing-tool integration is weak.** When it comes time to write papers, teams often move to tools like Overleaf, and the GitHub integration story is limited.

The article closes by saying that the system would continue to evolve and that suggestions were welcome. The goal in sharing the workflow was to help others adapt scientific research collaboration to remote work.
