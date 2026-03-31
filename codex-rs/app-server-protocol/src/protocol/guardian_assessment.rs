use crate::protocol::v2::CommandAction;
use codex_protocol::protocol::GuardianAssessmentAction;
use codex_protocol::protocol::GuardianAssessmentCommand;
use codex_protocol::protocol::GuardianAssessmentEvent;
use codex_shell_command::parse_command::parse_command;
use codex_shell_command::parse_command::shlex_join;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct GuardianCommandExecutionProjection {
    pub command: String,
    pub cwd: PathBuf,
    pub command_actions: Vec<CommandAction>,
}

pub fn guardian_command_execution_projection(
    assessment: &GuardianAssessmentEvent,
) -> Option<GuardianCommandExecutionProjection> {
    let action = assessment.action.as_ref()?;
    match action {
        GuardianAssessmentAction::Command(action)
            if matches!(action.tool.as_str(), "shell" | "exec_command") =>
        {
            let (command, command_actions) = match &action.command {
                GuardianAssessmentCommand::String(command) => (
                    command.clone(),
                    vec![CommandAction::Unknown {
                        command: command.clone(),
                    }],
                ),
                GuardianAssessmentCommand::Argv(argv) => {
                    let command = shlex_join(argv);
                    let parsed_cmd = parse_command(argv);
                    let command_actions = if parsed_cmd.is_empty() {
                        vec![CommandAction::Unknown {
                            command: command.clone(),
                        }]
                    } else {
                        parsed_cmd.into_iter().map(CommandAction::from).collect()
                    };
                    (command, command_actions)
                }
            };
            Some(GuardianCommandExecutionProjection {
                command,
                cwd: action.cwd.clone(),
                command_actions,
            })
        }
        #[cfg(unix)]
        GuardianAssessmentAction::Execve(action) => {
            let argv = if action.argv.is_empty() {
                vec![action.program.clone()]
            } else {
                std::iter::once(action.program.clone())
                    .chain(action.argv.iter().skip(1).cloned())
                    .collect::<Vec<_>>()
            };
            let command = shlex_join(&argv);
            let parsed_cmd = parse_command(&argv);
            let command_actions = if parsed_cmd.is_empty() {
                vec![CommandAction::Unknown {
                    command: command.clone(),
                }]
            } else {
                parsed_cmd.into_iter().map(CommandAction::from).collect()
            };
            Some(GuardianCommandExecutionProjection {
                command,
                cwd: action.cwd.clone(),
                command_actions,
            })
        }
        _ => None,
    }
}
