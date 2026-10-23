// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"context"
	"log/slog"

	"github.com/spf13/cobra"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var deleteCmd = &cobra.Command{
	Use:   "delete",
	Short: "Delete GCP resources",
	RunE: func(cmd *cobra.Command, args []string) error {
		return cmd.Help()
	},
}

func deleteClusterIdempotent(ctx context.Context, cfg *Config) error {
	if err := validateClusterLocation(cfg); err != nil {
		return err
	}
	err := deleteCluster(ctx, cfg)
	if status.Code(err) == codes.NotFound {
		slog.Info("Cluster does not exist, skipping deletion", slog.String("cluster", cfg.ClusterName))
		return nil
	}
	return err
}

var deleteClusterCmd = &cobra.Command{
	Use:   "cluster",
	Short: "Delete GKE cluster",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()
		if err := resolveProjectID(ctx, &cfg); err != nil {
			return err
		}
		return deleteClusterIdempotent(ctx, &cfg)
	},
}

func init() {
	rootCmd.AddCommand(deleteCmd)
	deleteCmd.AddCommand(deleteClusterCmd)
	deleteClusterCmd.Flags().StringVar(&cfg.ClusterName, "name", getEnv("CLUSTER_NAME", "substrate-poc"), "Name of the GKE cluster [env: CLUSTER_NAME]")
	deleteClusterCmd.Flags().StringVar(&cfg.ClusterLocation, "location", getEnv("CLUSTER_LOCATION", "us-west1-c"), "Zone or region for the cluster [env: CLUSTER_LOCATION]")
}
