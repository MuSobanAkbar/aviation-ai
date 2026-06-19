terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "us-east-1" 
}

# The Cloud Pantry for your Airline Policy PDFs
resource "aws_s3_bucket" "travel_agent_storage" {
  bucket = "soban-flight-agent-policies-2026" 
  
  tags = {
    ManagedBy   = "terraform"
    Project     = "flight-travel-agent"
    Environment = "Dev"
  }
}


output "policy_bucket_arn" {
  value       = aws_s3_bucket.travel_agent_storage.arn
  description = "The permanent cloud address of your storage bucket"
}