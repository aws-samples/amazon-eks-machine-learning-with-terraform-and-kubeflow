output "db_secret_arn" {
  description = "DB secret ARN"
  value = aws_rds_cluster.db.master_user_secret[0].secret_arn
}
