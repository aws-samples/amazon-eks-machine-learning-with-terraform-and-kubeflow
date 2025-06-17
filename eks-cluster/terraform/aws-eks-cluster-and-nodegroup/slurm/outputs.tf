output "db_password" {
  description = "DB password"
  sensitive   = true
  value = random_password.db.result
}
