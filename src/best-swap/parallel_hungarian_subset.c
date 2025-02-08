#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "parallel_hungarian.h"

#define INF (0x7FFFFFFF)
#define verbose (0)

#define hungarian_test_alloc(X) do {if ((void *)(X) == NULL) fprintf(stderr, "Out of memory in %s, (%s, line %d).\n", __FUNCTION__, __FILE__, __LINE__); } while (0)


void hungarian_print_matrixi(int** C, int rows, int cols) {
  int i,j;
  fprintf(stderr , "\n");
  for(i=0; i<rows; i++) {
    fprintf(stderr, " [");
    for(j=0; j<cols; j++) {
      fprintf(stderr, "%5d ",C[i][j]);
    }
    fprintf(stderr, "]\n");
  }
  fprintf(stderr, "\n");
}

void hungarian_print_matrixd(double** C, int rows, int cols) {
  int i,j;
  fprintf(stderr , "\n");
  for(i=0; i<rows; i++) {
    fprintf(stderr, " [");
    for(j=0; j<cols; j++) {
      fprintf(stderr, "%5f ",C[i][j]);
    }
    fprintf(stderr, "]\n");
  }
  fprintf(stderr, "\n");
}


void hungarian_print_assignment(hungarian_problem_t* p) {
  int i,j;
  for(i=0;i<p->num_rows;i++) {
	for(j=0; j<p->num_cols; j++) {
		if(p->assignment[i][j]==1) {printf("(%i,%i) ",i,j);}
	}
  }
  printf("\n");
}

int* return_assignment(hungarian_problem_t* p) {
    int* assignments = (int*)malloc(p->num_rows * sizeof(int));
    if (!assignments) {
        return NULL;
    }

    for (int i = 0; i < p->num_rows; i++) {
        assignments[i] = -1;
        for (int j = 0; j < p->num_cols; j++) {
            if (p->assignment[i][j] == HUNGARIAN_ASSIGNED) {
                assignments[i] = j;
                break;
            }
        }
    }

    return assignments;
}

void hungarian_print_costmatrix(hungarian_problem_t* p) {
  hungarian_print_matrixd(p->cost, p->num_rows, p->num_cols) ;
}

void hungarian_print_status(hungarian_problem_t* p) {
  fprintf(stderr,"cost:\n");
  hungarian_print_costmatrix(p);
  fprintf(stderr,"assignment:\n");
  hungarian_print_assignment(p);
}

double hungarian_imax(double a, double b) {
  return (a<b)?b:a;
}

int hungarian_init(hungarian_problem_t* p, double** cost_matrix, int rows, int cols, int mode) {
  int i,j, org_cols, org_rows;
  double max_cost = 0;
  
  org_cols = cols;
  org_rows = rows;

  rows = hungarian_imax(cols, rows);
  cols = rows;
  
  p->num_rows = rows;
  p->num_cols = cols;

  p->cost = (double**)calloc(rows,sizeof(double*));
  hungarian_test_alloc(p->cost);
  p->assignment = (int**)calloc(rows,sizeof(int*));
  hungarian_test_alloc(p->assignment);

  #pragma omp parallel for private(j) reduction(max:max_cost)
  for(i=0; i<p->num_rows; i++) {
    p->cost[i] = (double*)calloc(cols,sizeof(double));
    hungarian_test_alloc(p->cost[i]);
    p->assignment[i] = (int*)calloc(cols,sizeof(int));
    hungarian_test_alloc(p->assignment[i]);
    for(j=0; j<p->num_cols; j++) {
      p->cost[i][j] =  (i < org_rows && j < org_cols) ? cost_matrix[i][j] : 0;
      p->assignment[i][j] = 0;
      if (max_cost < p->cost[i][j])
        max_cost = p->cost[i][j];
    }
  }

  if (mode == HUNGARIAN_MODE_MAXIMIZE_UTIL) {
    #pragma omp parallel for private(j)
    for(i=0; i<p->num_rows; i++) {
      for(j=0; j<p->num_cols; j++) {
        p->cost[i][j] =  max_cost - p->cost[i][j];
      }
    }
  }
  else if (mode == HUNGARIAN_MODE_MINIMIZE_COST) {
    /* nothing to do */
  }
  else 
    fprintf(stderr,"%s: unknown mode. Mode was set to HUNGARIAN_MODE_MINIMIZE_COST !\n", __FUNCTION__);
  
  return rows;
}

void hungarian_free(hungarian_problem_t* p) {
  int i;
  for(i=0; i<p->num_rows; i++) {
    free(p->cost[i]);
    free(p->assignment[i]);
  }
  free(p->cost);
  free(p->assignment);
  p->cost = NULL;
  p->assignment = NULL;
}

void hungarian_solve(hungarian_problem_t* p) {
  int i, j, m, n, k, l, t, q, unmatched;
  float s,cost;
  int* col_mate;
  int* row_mate;
  int* parent_row;
  int* unchosen_row;
  float* row_dec;
  float* col_inc;
  float* slack;
  int* slack_row;

  cost=0;
  m =p->num_rows;
  n =p->num_cols;

  col_mate = (int*)calloc(m,sizeof(int));
  hungarian_test_alloc(col_mate);
  unchosen_row = (int*)calloc(m,sizeof(int));
  hungarian_test_alloc(unchosen_row);
  row_dec  = (float*)calloc(m,sizeof(float));
  hungarian_test_alloc(row_dec);
  slack_row  = (int*)calloc(n,sizeof(int));
  hungarian_test_alloc(slack_row);

  row_mate = (int*)calloc(n,sizeof(int));
  hungarian_test_alloc(row_mate);
  parent_row = (int*)calloc(n,sizeof(int));
  hungarian_test_alloc(parent_row);
  col_inc = (float*)calloc(n,sizeof(float));
  hungarian_test_alloc(col_inc);
  slack = (float*)calloc(n,sizeof(float));
  hungarian_test_alloc(slack);

  #pragma omp parallel for
  for (i=0;i<m;i++) {
    col_mate[i]=0;
    unchosen_row[i]=0;
    row_dec[i]=0.;
  }
  #pragma omp parallel for
  for (j=0;j<n;j++) {
    row_mate[j]=0;
    parent_row[j] = 0;
    col_inc[j]=0.;
    slack[j]=0;
  }

  #pragma omp parallel for private(j)
  for (i=0;i<m;++i)
    for (j=0;j<n;++j)
      p->assignment[i][j]=HUNGARIAN_NOT_ASSIGNED;

  // Subtract column minima
  #pragma omp parallel for private(k) reduction(+:cost)
  for (l=0;l<n;l++) {
    double s = p->cost[0][l];
    for (k=1;k<m;k++) 
      if (p->cost[k][l]<s)
        s=p->cost[k][l];
    cost+=s;
    if (s!=0)
      for (k=0;k<m;k++)
        p->cost[k][l]-=s;
  }

  // Initialize state variables
  #pragma omp parallel for
  for (l=0;l<n;l++) {
    row_mate[l]= -1;
    parent_row[l]= -1;
    col_inc[l]=0;
    slack[l]=INF;
  }

  t=0;
  for (k=0;k<m;k++) {
    s=p->cost[k][0];
    for (l=1;l<n;l++)
      if (p->cost[k][l]<s)
        s=p->cost[k][l];
    row_dec[k]=s;
    for (l=0;l<n;l++)
      if (s==p->cost[k][l] && row_mate[l]<0) {
        col_mate[k]=l;
        row_mate[l]=k;
        goto row_done;
      }
    col_mate[k]= -1;
    unchosen_row[t++]=k;
  row_done: ;
  }

  if (t==0) goto done;
  unmatched=t;
  
  while (1) {
    q=0;
    while (1) {
      while (q<t) {
        k=unchosen_row[q];
        s=row_dec[k];
        for (l=0;l<n;l++) {
          if (slack[l]) {
            float del = p->cost[k][l]-s+col_inc[l];
            if (del<slack[l]) {
              if (del==0) {
                if (row_mate[l]<0) goto breakthru;
                slack[l]=0;
                parent_row[l]=k;
                unchosen_row[t++]=row_mate[l];
              } else {
                slack[l]=del;
                slack_row[l]=k;
              }
            }
          }
        }
        q++;
      }

      s=INF;
      #pragma omp parallel for reduction(min:s)
      for (l=0;l<n;l++) {
        if (slack[l] && slack[l]<s) s=slack[l];
      }

      #pragma omp parallel for
      for (q=0;q<t;q++)
        row_dec[unchosen_row[q]]+=s;

      for (l=0;l<n;l++) {
        if (slack[l]) {
          slack[l]-=s;
          if (slack[l]==0.) {
            k=slack_row[l];
            if (row_mate[l]<0) {
              for (j=l+1;j<n;j++)
                if (slack[j]==0) col_inc[j]+=s;
              goto breakthru;
            } else {
              parent_row[l]=k;
              unchosen_row[t++]=row_mate[l];
            }
          }
        } else {
          col_inc[l]+=s;
        }
      }
    }
  breakthru:
    while (1) {
      j=col_mate[k];
      col_mate[k]=l;
      row_mate[l]=k;
      if (j<0) break;
      k=parent_row[j];
      l=j;
    }
    if (--unmatched==0) break;
    t=0;
    #pragma omp parallel for
    for (l=0;l<n;l++) {
      parent_row[l]= -1;
      slack[l]=INF;
    }
    for (k=0;k<m;k++)
      if (col_mate[k]<0)
        unchosen_row[t++]=k;
  }

done:
  // Final assignment and cost updates
  #pragma omp parallel for
  for (i=0;i<m;++i)
    p->assignment[i][col_mate[i]]=HUNGARIAN_ASSIGNED;

  #pragma omp parallel for private(l)
  for (k=0;k<m;++k)
    for (l=0;l<n;++l)
      p->cost[k][l]=p->cost[k][l]-row_dec[k]+col_inc[l];

  free(slack);
  free(col_inc);
  free(parent_row);
  free(row_mate);
  free(slack_row);
  free(row_dec);
  free(unchosen_row);
  free(col_mate);
}