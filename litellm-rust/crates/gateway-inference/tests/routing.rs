mod support;

use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn message() -> Value {
    json!({"id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}})
}

async fn upstream(status: u16) -> MockServer {
    let server = MockServer::start().await;
    let template = match status {
        200 => ResponseTemplate::new(200).set_body_json(message()),
        status => ResponseTemplate::new(status).set_body_json(
            json!({"type": "error", "error": {"type": "api_error", "message": "upstream failed"}}),
        ),
    };
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(template)
        .mount(&server)
        .await;
    server
}

async fn hits(server: &MockServer) -> usize {
    server.received_requests().await.unwrap().len()
}

const REQUESTS: usize = 8;

#[rstest]
#[case::messages("/v1/messages")]
#[case::chat("/v1/chat/completions")]
#[tokio::test]
async fn a_group_retries_failures_on_its_healthy_deployment(
    #[case] endpoint: &str,
    #[values(429, 500, 503)] status: u16,
) {
    let failing = upstream(status).await;
    let healthy = upstream(200).await;
    let app = support::app_with_group("anthropic/test-model", &[&failing.uri(), &healthy.uri()]);
    for _ in 0..REQUESTS {
        let response = support::post(
            app.clone(),
            endpoint,
            json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16}),
        )
        .await;
        assert_eq!(response.status(), 200);
    }
    assert_eq!(hits(&healthy).await, REQUESTS);
    assert!(hits(&failing).await <= REQUESTS);
}

#[rstest]
#[case::messages("/v1/messages")]
#[case::chat("/v1/chat/completions")]
#[tokio::test]
async fn a_group_whose_deployments_all_fail_returns_the_last_failure_after_trying_each(
    #[case] endpoint: &str,
) {
    let first = upstream(500).await;
    let second = upstream(500).await;
    let app = support::app_with_group("anthropic/test-model", &[&first.uri(), &second.uri()]);
    let response = support::post(
        app,
        endpoint,
        json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16}),
    )
    .await;
    assert_eq!(response.status(), 500);
    assert_eq!((hits(&first).await, hits(&second).await), (1, 1));
}

#[rstest]
#[tokio::test]
async fn a_caller_error_is_not_retried() {
    let first = upstream(400).await;
    let second = upstream(400).await;
    let app = support::app_with_group("anthropic/test-model", &[&first.uri(), &second.uri()]);
    let response = support::post(
        app,
        "/v1/messages",
        json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16}),
    )
    .await;
    assert_eq!(response.status(), 400);
    assert_eq!(hits(&first).await + hits(&second).await, 1);
}

async fn openai_upstream(status: u16, streaming: bool) -> MockServer {
    let server = MockServer::start().await;
    let template = match (status, streaming) {
        (200, false) => ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test", "object": "chat.completion", "model": "gpt-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"}],
        })),
        (200, true) => ResponseTemplate::new(200).set_body_raw(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n",
            "text/event-stream",
        ),
        (status, _) => ResponseTemplate::new(status).set_body_json(
            json!({"error": {"message": "upstream failed", "type": "server_error"}}),
        ),
    };
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(template)
        .mount(&server)
        .await;
    server
}

#[rstest]
#[tokio::test]
async fn openai_passthrough_retries_failures_on_its_healthy_deployment(
    #[values(429, 500)] status: u16,
    #[values(false, true)] streaming: bool,
) {
    let failing = openai_upstream(status, streaming).await;
    let healthy = openai_upstream(200, streaming).await;
    let bases = [
        format!("{}/v1", failing.uri()),
        format!("{}/v1", healthy.uri()),
    ];
    let app = support::app_with_group("openai/gpt-test", &[&bases[0], &bases[1]]);
    for _ in 0..REQUESTS {
        let response = support::post(
            app.clone(),
            "/v1/chat/completions",
            json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
                "stream": streaming}),
        )
        .await;
        assert_eq!(response.status(), 200);
    }
    assert_eq!(hits(&healthy).await, REQUESTS);
}

#[rstest]
#[case::messages("/v1/messages")]
#[case::chat("/v1/chat/completions")]
#[tokio::test]
async fn metrics_label_the_deployment_that_served_the_request(#[case] endpoint: &str) {
    let failing = upstream(500).await;
    let healthy = upstream(200).await;
    let app = support::app_with_group("anthropic/test-model", &[&failing.uri(), &healthy.uri()]);
    let (response, resolved) = litellm_gateway_inference::observe::scope(support::post(
        app,
        endpoint,
        json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16}),
    ))
    .await;
    assert_eq!(response.status(), 200);
    let resolved = resolved.unwrap();
    assert_eq!(resolved.model_group, "public/model");
    assert_eq!(resolved.api_base, Some(healthy.uri()));
}
