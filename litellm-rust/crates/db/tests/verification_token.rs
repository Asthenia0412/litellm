use std::time::{Duration, SystemTime};

use litellm_db::{Restrictions, Team, User, VerificationToken, find_verification_token};
use litellm_db_testing::MigratedPostgres;
use rstest::rstest;
use serde_json::json;

const FIXTURES: &str = r#"
INSERT INTO "LiteLLM_TeamTable" (team_id, blocked, models, spend, max_budget, metadata, tpm_limit,
                                 model_id, created_at, updated_at)
  VALUES ('team-1', true, ARRAY['gpt-a'], 2.5, 10.0, '{"require_trace_id": true}', 100, NULL,
          now(), now()),
         ('team-2', false, ARRAY[]::text[], 0, NULL, '{}', NULL, NULL, now(), now());
INSERT INTO "LiteLLM_UserTable" (user_id, models, rpm_limit, max_budget, policies)
  VALUES ('user-1', ARRAY['gpt-b'], 5, 1.0, ARRAY['p']),
         ('user-2', ARRAY[]::text[], NULL, NULL, ARRAY[]::text[]);
INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by) VALUES ('b-1', 'x', 'x');
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id) VALUES ('user-2', 'team-2', 'b-1');
INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, created_by, updated_by)
  VALUES ('org-1', 'org', 'b-1', 'x', 'x');
INSERT INTO "LiteLLM_ProjectTable" (project_id, created_by, updated_by) VALUES ('project-1', 'x', 'x');
INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id) VALUES ('op-1');
INSERT INTO "LiteLLM_VerificationToken" (token, expires, blocked, models) VALUES
  ('active', NULL, NULL, ARRAY[]::text[]),
  ('blocked', NULL, true, ARRAY[]::text[]),
  ('no-models', NULL, NULL, NULL);
INSERT INTO "LiteLLM_VerificationToken"
  (token, expires, blocked, models, spend, max_budget, allowed_routes, key_alias, key_name, team_id,
   user_id, metadata)
  VALUES ('scoped', '2030-01-02 03:04:05.678', false, ARRAY['gpt-a', 'claude-*'], 1.5, 3.0,
          ARRAY['llm_api_routes'], 'prod', 'sk-...abcd', 'team-1', 'user-1', '{"tags": ["a"]}'),
         ('orphan', NULL, false, ARRAY[]::text[], 0, NULL, ARRAY[]::text[], NULL, NULL, 'gone',
          NULL, '{}'),
         ('member', NULL, false, ARRAY[]::text[], 0, NULL, ARRAY[]::text[], NULL, NULL, 'team-2',
          'user-2', '{}');
INSERT INTO "LiteLLM_VerificationToken"
  (token, models, rpm_limit, budget_id, policies, aliases, allowed_cache_controls, project_id,
   organization_id, agent_id, object_permission_id)
  VALUES ('restricted', ARRAY[]::text[], 10, 'b-1', ARRAY['p'], '{"a": "b"}', ARRAY['no-cache'],
          'project-1', 'org-1', 'agent-1', 'op-1');
INSERT INTO "LiteLLM_VerificationToken" (token, models, max_parallel_requests)
  VALUES ('limited-only', ARRAY[]::text[], 2);
"#;

#[rstest]
#[tokio::test]
async fn reads_the_gating_columns_of_a_verification_token_and_its_owners() {
    let postgres = MigratedPostgres::start().await.unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "test fixture rows; the lookup under test uses the checked macro"
    )]
    sqlx::raw_sql(FIXTURES)
        .execute(postgres.pool())
        .await
        .unwrap();
    let pool = postgres.pool();
    let find = |token: &'static str| async move {
        find_verification_token(pool, token).await.unwrap().unwrap()
    };

    assert_eq!(
        find("active").await,
        VerificationToken {
            token: "active".into(),
            expires: None,
            blocked: false,
            models: vec![],
            spend: 0.0,
            max_budget: None,
            allowed_routes: vec![],
            key_alias: None,
            key_name: None,
            metadata: json!({}),
            restrictions: Restrictions::default(),
            team_id: None,
            team: None,
            team_member_budget: false,
            user_id: None,
            user: None,
        }
    );
    assert_eq!(
        find("scoped").await,
        VerificationToken {
            token: "scoped".into(),
            // 2030-01-02T03:04:05.678Z
            expires: Some(SystemTime::UNIX_EPOCH + Duration::from_millis(1_893_553_445_678)),
            blocked: false,
            models: vec!["gpt-a".into(), "claude-*".into()],
            spend: 1.5,
            max_budget: Some(3.0),
            allowed_routes: vec!["llm_api_routes".into()],
            key_alias: Some("prod".into()),
            key_name: Some("sk-...abcd".into()),
            metadata: json!({"tags": ["a"]}),
            restrictions: Restrictions::default(),
            team_id: Some("team-1".into()),
            team: Some(Team {
                blocked: true,
                models: vec!["gpt-a".into()],
                spend: 2.5,
                max_budget: Some(10.0),
                metadata: json!({"require_trace_id": true}),
                restrictions: Restrictions {
                    rate_limits: true,
                    ..Restrictions::default()
                },
            }),
            team_member_budget: false,
            user_id: Some("user-1".into()),
            user: Some(User {
                models: vec!["gpt-b".into()],
                restrictions: Restrictions {
                    rate_limits: true,
                    budgets: true,
                    policies: true,
                    ..Restrictions::default()
                },
            }),
        }
    );
    assert_eq!(
        find("restricted").await.restrictions,
        Restrictions {
            rate_limits: true,
            budgets: true,
            policies: true,
            model_aliases: true,
            object_permissions: true,
            cache_controls: true,
            project: true,
            organization: true,
            agent: true,
        }
    );
    assert_eq!(
        find("limited-only").await.restrictions,
        Restrictions {
            rate_limits: true,
            ..Restrictions::default()
        }
    );
    let member = find("member").await;
    assert!(member.team_member_budget);
    assert_eq!(member.team.unwrap().restrictions, Restrictions::default());
    assert_eq!(member.user.unwrap().restrictions, Restrictions::default());
    let orphan = find("orphan").await;
    assert_eq!(
        (orphan.team_id.as_deref(), orphan.team),
        (Some("gone"), None)
    );
    assert!(find("blocked").await.blocked);
    assert_eq!(find("no-models").await.models, Vec::<String>::new());
    assert_eq!(
        find_verification_token(pool, "missing").await.unwrap(),
        None
    );
}
