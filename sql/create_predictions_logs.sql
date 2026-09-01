CREATE DATABASE IF NOT EXISTS ticket_classification
CHARACTER SET utf8mb4;

USE ticket_classification;


CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    text VARCHAR(200) NOT NULL,
    predicted_label VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    latency_ms FLOAT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX idx_created_at
ON prediction_logs(created_at);